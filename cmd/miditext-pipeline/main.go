package main

import (
	"crypto/md5"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"gitlab.com/gomidi/midi/v2/smf"
	"zenmidi/pkg/miditext"
)

type Config struct {
	InputDir    string
	OutputDir   string
	Workers     int
	MinNotes    int
	MaxNotes    int
	MinBars     int
	MaxBars     int
	MaxChannels int
}

type Stats struct {
	TotalFiles     int64
	MidiReadErrors int64
	ParsedFiles    int64
	FilterChannel  int64
	FilterNotes    int64
	FilterBars     int64
	DecompileError int64
	CompileError   int64
	DuplicateCount int64
	SavedCount     int64
}

func main() {
	var cfg Config
	flag.StringVar(&cfg.InputDir, "input", "", "Directory containing MIDI files to process")
	flag.StringVar(&cfg.OutputDir, "output", "data/raw", "Directory to save decompiled MIDIText files")
	flag.IntVar(&cfg.Workers, "workers", 16, "Number of concurrent worker goroutines")
	flag.IntVar(&cfg.MinNotes, "min-notes", 16, "Minimum note count to accept")
	flag.IntVar(&cfg.MaxNotes, "max-notes", 10000, "Maximum note count to accept")
	flag.IntVar(&cfg.MinBars, "min-bars", 4, "Minimum bar count to accept")
	flag.IntVar(&cfg.MaxBars, "max-bars", 300, "Maximum bar count to accept")
	flag.IntVar(&cfg.MaxChannels, "max-channels", 4, "Maximum active MIDI channels to accept (e.g. 1 for solo, 2 for piano, 4 for celtic band)")
	flag.Parse()

	if cfg.InputDir == "" {
		log.Fatal("Error: -input directory must be specified")
	}

	err := os.MkdirAll(cfg.OutputDir, 0755)
	if err != nil {
		log.Fatalf("Error creating output directory: %v", err)
	}

	// 1. Gather all MIDI files
	var files []string
	err = filepath.WalkDir(cfg.InputDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			ext := strings.ToLower(filepath.Ext(path))
			if ext == ".mid" || ext == ".midi" {
				files = append(files, path)
			}
		}
		return nil
	})
	if err != nil {
		log.Fatalf("Error walking input directory: %v", err)
	}

	totalFiles := len(files)
	fmt.Printf("Found %d MIDI files to process.\n", totalFiles)
	if totalFiles == 0 {
		return
	}

	// Channel for feeding files to workers
	fileChan := make(chan string, totalFiles)
	for _, f := range files {
		fileChan <- f
	}
	close(fileChan)

	var stats Stats
	stats.TotalFiles = int64(totalFiles)

	// Deduplication map protection
	var dedupMu sync.Mutex
	seenHashes := make(map[string]bool)

	// Worker pool
	var wg sync.WaitGroup
	startTime := time.Now()

	for i := 0; i < cfg.Workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range fileChan {
				processFile(path, &cfg, &stats, &dedupMu, seenHashes)
			}
		}()
	}

	// Progress monitor
	go func() {
		for {
			time.Sleep(2 * time.Second)
			parsed := atomic.LoadInt64(&stats.ParsedFiles)
			saved := atomic.LoadInt64(&stats.SavedCount)
			if parsed >= int64(totalFiles) {
				break
			}
			fmt.Printf("Progress: %d/%d processed (%d saved)...\n", parsed, totalFiles, saved)
		}
	}()

	wg.Wait()
	elapsed := time.Since(startTime)

	fmt.Println("\nProcessing Complete!")
	fmt.Printf("Time taken: %v\n", elapsed)
	fmt.Printf("Total Files: %d\n", stats.TotalFiles)
	fmt.Printf("MIDI Read/Parse Errors: %d\n", stats.MidiReadErrors)
	fmt.Printf("Filtered (Too many channels): %d\n", stats.FilterChannel)
	fmt.Printf("Filtered (Too few/many notes): %d\n", stats.FilterNotes)
	fmt.Printf("Filtered (Too few/many bars): %d\n", stats.FilterBars)
	fmt.Printf("Decompile Errors: %d\n", stats.DecompileError)
	fmt.Printf("Compile Validation Errors: %d\n", stats.CompileError)
	fmt.Printf("Duplicate Files: %d\n", stats.DuplicateCount)
	fmt.Printf("Successfully Saved: %d\n", stats.SavedCount)
}

func processFile(path string, cfg *Config, stats *Stats, dedupMu *sync.Mutex, seenHashes map[string]bool) {
	defer atomic.AddInt64(&stats.ParsedFiles, 1)
	defer func() {
		if r := recover(); r != nil {
			atomic.AddInt64(&stats.MidiReadErrors, 1)
		}
	}()

	// 1. Read MIDI metadata for filtering
	data, err := os.ReadFile(path)
	if err != nil {
		atomic.AddInt64(&stats.MidiReadErrors, 1)
		return
	}

	s, err := smf.ReadFrom(strings.NewReader(string(data)))
	if err != nil {
		atomic.AddInt64(&stats.MidiReadErrors, 1)
		return
	}

	// Extract time resolution & properties
	ppqVal, ok := s.TimeFormat.(smf.MetricTicks)
	if !ok {
		atomic.AddInt64(&stats.MidiReadErrors, 1)
		return
	}
	ppq := int(ppqVal)

	// Count channels, notes, bars
	var noteCount int
	var maxTick int
	activeChannels := make(map[uint8]bool)

	for _, track := range s.Tracks {
		absTick := 0
		for _, ev := range track {
			absTick += int(ev.Delta)
			if absTick > maxTick {
				maxTick = absTick
			}
			var channel, key, vel uint8
			if ev.Message.GetNoteOn(&channel, &key, &vel) && vel > 0 {
				noteCount++
				activeChannels[channel] = true
			}
		}
	}

	// Filter 1: Channels count
	if len(activeChannels) == 0 || len(activeChannels) > cfg.MaxChannels {
		atomic.AddInt64(&stats.FilterChannel, 1)
		return
	}

	// Filter 2: Note count
	if noteCount < cfg.MinNotes || noteCount > cfg.MaxNotes {
		atomic.AddInt64(&stats.FilterNotes, 1)
		return
	}

	// Filter 3: Bar count
	// Assuming 4/4 as fallback unless TS is found
	beatsPerBar := 4
	for _, track := range s.Tracks {
		for _, ev := range track {
			var num, denom, clocks, notes32 uint8
			if ev.Message.GetMetaTimeSig(&num, &denom, &clocks, &notes32) {
				beatsPerBar = int(num)
				break
			}
		}
	}

	barTicks := ppq * beatsPerBar
	if barTicks == 0 {
		barTicks = 480 * 4
	}
	barCount := (maxTick + barTicks - 1) / barTicks

	if barCount < cfg.MinBars || barCount > cfg.MaxBars {
		atomic.AddInt64(&stats.FilterBars, 1)
		return
	}

	// 2. Decompile to MIDIText
	miditextStr, err := miditext.Decompile(data)
	if err != nil {
		atomic.AddInt64(&stats.DecompileError, 1)
		return
	}

	// 3. Compile validation (round-trip check)
	_, err = miditext.Compile(miditextStr)
	if err != nil {
		atomic.AddInt64(&stats.CompileError, 1)
		return
	}

	// 4. Content-hash deduplication (MD5 of the output text)
	hasher := md5.New()
	io.WriteString(hasher, miditextStr)
	hashStr := hex.EncodeToString(hasher.Sum(nil))

	dedupMu.Lock()
	if seenHashes[hashStr] {
		dedupMu.Unlock()
		atomic.AddInt64(&stats.DuplicateCount, 1)
		return
	}
	seenHashes[hashStr] = true
	dedupMu.Unlock()

	// 5. Save the output
	// Keep directory hierarchy under OutputDir
	rel, err := filepath.Rel(cfg.InputDir, path)
	if err != nil {
		rel = filepath.Base(path)
	}
	outPath := filepath.Join(cfg.OutputDir, rel)
	outPath = outPath[:len(outPath)-len(filepath.Ext(outPath))] + ".txt"

	err = os.MkdirAll(filepath.Dir(outPath), 0755)
	if err != nil {
		return
	}

	err = os.WriteFile(outPath, []byte(miditextStr), 0644)
	if err != nil {
		return
	}

	atomic.AddInt64(&stats.SavedCount, 1)
}

package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"zenmidi/pkg/miditext"
)

func main() {
	var input string

	if len(os.Args) > 1 && os.Args[1] != "-" {
		// Read from file path
		data, err := os.ReadFile(os.Args[1])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
			os.Exit(1)
		}
		input = string(data)
	} else {
		// Read from stdin
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading stdin: %v\n", err)
			os.Exit(1)
		}
		input = string(data)
	}

	input = strings.TrimSpace(input)
	if input == "" {
		fmt.Fprintln(os.Stderr, "Error: empty input")
		os.Exit(1)
	}

	_, err := miditext.Compile(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Validation Error: %v\n", err)
		os.Exit(1)
	}

	// Success
	os.Exit(0)
}

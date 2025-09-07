"""test_parse.py
Quick test runner for input parsing.
Tries to import parse_input from core.parsing, falls back to anki_csv_builder.parse_input.
Usage: python test_parse.py
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

# Try to import parse_input from core.parsing first, fall back to anki_csv_builder
try:
    from core.parsing import parse_input  # type: ignore
    SOURCE = "core.parsing"
except Exception:
    try:
        from anki_csv_builder import parse_input  # type: ignore
        SOURCE = "anki_csv_builder.parse_input"
    except Exception as e:
        print("Error: no parse_input function available. Please create core/parsing.py or ensure anki_csv_builder.parse_input is present.")
        print(e)
        sys.exit(1)

INPUT_DIR = Path("tests/test_inputs")
if not INPUT_DIR.exists():
    print(f"Input dir {INPUT_DIR} does not exist.")
    sys.exit(1)

print(f"Using parse_input from: {SOURCE}\n")

for fp in sorted(INPUT_DIR.iterdir()):
    if fp.is_file():
        print(f"=== {fp.name} ===")
        try:
            text = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = fp.read_text(encoding="utf-16")
            except Exception as e:
                print(f"Failed to read {fp}: {e}")
                continue
        rows = []
        try:
            rows = parse_input(text)
        except Exception as e:
            print(f"parse_input raised an exception for {fp.name}: {e}")
            continue
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        print()

print("Done.")

#!/usr/bin/env bash

OUTPUT_DIR="checksums"
OUTPUT_FILE="$OUTPUT_DIR/checksums.txt"

mkdir -p "$OUTPUT_DIR"

project_name=$(basename "$(git rev-parse --show-toplevel)")
current_date=$(date +"%Y-%m-%d %H:%M:%S")

echo "# $project_name $current_date" > "$OUTPUT_FILE"

git ls-files -z | grep -z -v  -e '^\.gitignore$' -e "^${OUTPUT_FILE}" | xargs -0 shasum -a 256 >> "$OUTPUT_FILE"
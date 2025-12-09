#!/usr/bin/env bash

CHECKSUM_FILE="checksums/checksums.txt"

while IFS= read -r line; do
    [ -z "$line" ] && continue
    echo "$line" | grep -qE '^[[:space:]]*#' && continue

    sum="${line%% *}"

    filepath="${line#*$sum }"

    filepath="${filepath#"${filepath%%[![:space:]]*}"}"

    if [ ! -f "$filepath" ]; then
        echo "$filepath: absent"
        continue
    fi

    current_sum=$(shasum -a 256 "$filepath" | awk '{print $1}')

    if [ "$current_sum" = "$sum" ]; then
        echo "$filepath: OK"
    else
        echo "$filepath: changed"
    fi

done < "$CHECKSUM_FILE"
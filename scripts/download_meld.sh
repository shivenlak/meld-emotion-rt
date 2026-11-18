#!/usr/bin/env bash
# Download the official MELD raw release (~10.9 GB) and extract it to data/MELD.Raw.
# The server is slow per connection (~2 MB/s), so we fetch 8 byte ranges in
# parallel and join them. Re-running resumes any partial parts.
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
OUT=data/download
mkdir -p "$OUT"
TAR="$OUT/MELD.Raw.tar.gz"
N=8

if [ ! -f "$TAR.done" ]; then
  SIZE=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  CHUNK=$(( (SIZE + N - 1) / N ))
  for i in $(seq 0 $((N-1))); do
    start=$(( i * CHUNK )); end=$(( start + CHUNK - 1 )); [ $end -ge $SIZE ] && end=$((SIZE-1))
    part="$OUT/part$i"; have=0; [ -f "$part" ] && have=$(stat -f%z "$part" 2>/dev/null || stat -c%s "$part")
    want=$(( end - start + 1 ))
    if [ "$have" -lt "$want" ]; then
      curl -sL -r "$((start+have))-$end" "$URL" >> "$part" &
    fi
  done
  wait
  cat $(for i in $(seq 0 $((N-1))); do echo "$OUT/part$i"; done) > "$TAR"
  [ "$(stat -f%z "$TAR" 2>/dev/null || stat -c%s "$TAR")" -eq "$SIZE" ] || { echo "size mismatch"; exit 1; }
  rm -f "$OUT"/part*; touch "$TAR.done"
fi

# The outer archive contains MELD.Raw/{train,dev,test}.tar.gz plus the CSVs.
tar -xzf "$TAR" -C data
cd data/MELD.Raw
for s in train dev test; do [ -f "$s.tar.gz" ] && tar -xzf "$s.tar.gz" && rm "$s.tar.gz"; done
ls
echo "MELD extracted to data/MELD.Raw (you can delete $TAR to free ~11 GB)"

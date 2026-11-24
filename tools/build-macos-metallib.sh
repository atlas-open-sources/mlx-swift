#!/bin/zsh
set -euo pipefail
source_dir="${1:A}"
output="${2:A}"
temporary="$(mktemp -d "${output:h}/metal.XXXXXXXX")"
trap 'rm -r -- "$temporary"' EXIT

sources=("$source_dir"/**/*.metal(.N))
(( ${#sources} )) || { echo "No generated Metal kernels in $source_dir" >&2; exit 1; }
airs=()
for source in "${sources[@]}"; do
    relative="${source#$source_dir/}"
    air="$temporary/${${relative//\//_}%.metal}.air"
    xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math \
        -Wno-c++17-extensions -Wno-c++20-extensions -mmacosx-version-min=14.0 \
        -c "$source" -I "$source_dir" -o "$air"
    airs+=("$air")
done
xcrun -sdk macosx metallib "${airs[@]}" -o "$temporary/default.metallib"
mv "$temporary/default.metallib" "$output"

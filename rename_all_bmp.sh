#!/bin/bash

# 파일명이 공백을 포함한 BMP 파일을 밑줄(_)로 변경
folders=(
    "/workspace/dataset_split/train/bad"
    "/workspace/dataset_split/train/good"
    "/workspace/dataset_split/val/bad"
    "/workspace/dataset_split/val/good"
    "/workspace/dataset_split/test/bad"
    "/workspace/dataset_split/test/good"
)

for dir in "${folders[@]}"; do
    echo "🔍 Processing: $dir"
    find "$dir" -type f -name "*.bmp" | while read -r file; do
        if [[ "$file" =~ \  ]]; then
            newfile="${file// /_}"
            echo "👉 Rename: $file"
            echo "   to:     $newfile"
            mv "$file" "$newfile"
        fi
    done
done

echo "✅ 공백 파일명 치환 완료!"

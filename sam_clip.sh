#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <img_path> <mask_output_dir>"
  exit 1
fi

echo "img_path: $1"
echo "mask_output_dir: $2"

# Inference.py 실행 시간 측정
echo "Running Inference.py..."
start_time=$(date +%s.%N)
python Inference.py --img_path "$1" >/dev/null 2>&1
end_time=$(date +%s.%N)
inference_duration=$(echo "$end_time - $start_time" | bc)
printf "Inference.py completed in %.2f seconds.\n" "$inference_duration"

# sam_clip.py 실행 시간 측정
echo "Running sam_clip.py..."
start_time=$(date +%s.%N)
python sam_clip.py --mask_output_dir "$2" >/dev/null 2>&1
end_time=$(date +%s.%N)
sam_clip_duration=$(echo "$end_time - $start_time" | bc)
printf "sam_clip.py completed in %.2f seconds.\n" "$sam_clip_duration"

# 전체 실행 시간 출력 (선택 사항)
total_duration=$(echo "$inference_duration + $sam_clip_duration" | bc)
printf "Total execution time: %.2f seconds.\n" "$total_duration"

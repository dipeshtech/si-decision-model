#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p pretrained
revision=30b0a37ccaaa32f332884b96992754e246e48c5f
for file in config.json vocab.txt README.md model.safetensors; do
  curl --fail --location --retry 2 --output "pretrained/$file" "https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/$revision/$file"
done

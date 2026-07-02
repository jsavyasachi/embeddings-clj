#!/bin/sh

set -u

base_url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
fixture_dir="fixtures/all-MiniLM-L6-v2"

download() {
  url=$1
  dest=$2

  if [ -s "$dest" ]; then
    printf '%s\n' "Already present: $dest"
    return 0
  fi

  mkdir -p "$(dirname "$dest")" || {
    printf '%s\n' "Failed to create directory for $dest" >&2
    return 1
  }

  tmp="${dest}.tmp.$$"
  printf '%s\n' "Downloading $url -> $dest"
  if curl -fL --progress-bar -o "$tmp" "$url"; then
    if [ ! -s "$tmp" ]; then
      rm -f "$tmp"
      printf '%s\n' "Downloaded empty file for $dest" >&2
      return 1
    fi
    mv "$tmp" "$dest" || {
      rm -f "$tmp"
      printf '%s\n' "Failed to move downloaded file into place: $dest" >&2
      return 1
    }
  else
    status=$?
    rm -f "$tmp"
    printf '%s\n' "Failed to download $url" >&2
    return "$status"
  fi
}

download "$base_url/onnx/model.onnx" "$fixture_dir/model.onnx" || exit 1
download "$base_url/tokenizer.json" "$fixture_dir/tokenizer.json" || exit 1

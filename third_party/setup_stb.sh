#!/bin/bash

# Script to download stb_image_write.h for PNG output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STB_DIR="$SCRIPT_DIR/stb"

echo "Setting up stb_image_write library..."

mkdir -p "$STB_DIR"

echo "Downloading stb_image_write.h..."
wget -O "$STB_DIR/stb_image_write.h" \
    https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

echo ""
echo "stb_image_write.h downloaded successfully to:"
echo "  $STB_DIR/stb_image_write.h"

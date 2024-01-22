#!/bin/bash

cd "$(dirname "$0")"

# Find and delete all __pycache__ directories in the churninator directory
find ../churninator -type d -name "__pycache__" -exec rm -r {} +
rm -r ../churninator/models ../churninator/figures 2>/dev/null

echo "Deleted all __pycache__ directories"
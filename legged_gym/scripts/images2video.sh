#!/bin/bash

# Check if the user provided a path
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/main/folder"
    exit 1
fi

main_folder="$1"

# Check if the provided path exists
if [ ! -d "$main_folder" ]; then
    echo "Error: Directory '$main_folder' does not exist."
    exit 1
fi

# Loop through subfolders in the given path
for folder in "$main_folder"/*/; do
    if [ -d "$folder" ]; then
        output_video="${folder%/}.mp4"
        output_video="${output_video##*/}.mp4"  # Extracts only the folder name for the video

        echo "Processing $folder -> $output_video"

        # Run ffmpeg to create video
        ffmpeg -framerate 30 -pattern_type glob -i "$folder/*.png" -c:v libx264 -pix_fmt yuv420p "$main_folder/$output_video"
    fi
done

#!/bin/bash

# Convert video to H.264 with target specs
ffmpeg -i send1.mp4 \
    -vf "fps=25" \
    -c:v libx264 \
    -crf 23 \
    -preset medium \
    -c:a copy \
    output.mp4

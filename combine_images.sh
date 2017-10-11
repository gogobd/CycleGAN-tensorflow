#!/bin/bash

mkdir -p _videos
ffmpeg -r 30 -f image2 -s 852x480 -i _frames/$1/$1.%08d.jpg -vcodec libx264 -crf 20 -pix_fmt yuv420p _videos/$1-new.mp4

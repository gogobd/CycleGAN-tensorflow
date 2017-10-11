#!/bin/bash

mkdir -p _frames/$1
# ffmpeg -i _videos/$1 -vf fps=1/5 _frames/$1/$1.%08d.jpg
ffmpeg -i _videos/$1 -vf scale=256:256 -qscale:v 2 _frames/$1/$1.%08d.jpg

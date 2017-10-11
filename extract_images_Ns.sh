#!/bin/bash

mkdir -p _frames/$1
ffmpeg -i _videos/$1 -vf "fps=1/$2, scale=256:256" -qscale:v 2 _frames/$1/$1.%08d.jpg

ffmpeg -f image2 -s 256x256 -i -i _frames/$1/$1.%08d.jpg -vf scale=320:240 output_320x240.png
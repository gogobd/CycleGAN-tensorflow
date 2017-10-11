mkdir _frames\%1
"E:\Program Files\ffmpeg-12a419d-win64\bin\ffmpeg.exe" -i _videos\%1 -vf "fps=1/%2, scale=512:512" -qscale:v 2 _frames\%1\%1.%%08d.jpg

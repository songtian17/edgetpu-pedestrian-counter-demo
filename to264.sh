#!/bin/bash

input=$1
output=$2
ffmpeg -i $input -vcodec libx264 $output
echo "Converted video to H264."

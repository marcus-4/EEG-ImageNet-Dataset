#!/bin/bash

CLIP_SCRIPT="../src/blip_clip.py"
IMG_SCRIPT="../src/gen_img_list.py"

DATA_DIR="../data/"
G_OPTION="all"
# G_OPTION="coarse"
M_OPTION="eegnet"
B_OPTION=80
S_OPTION=0
P_OPTION="eegnet_s${S_OPTION}_1x_0.pth"
O_OPTION="../output/"


python $IMG_SCRIPT -d "../data/" -g "coarse" -m "svm" -b $B_OPTION -s 1 -o "../output/"
#python $CLIP_SCRIPT -d "../data/" -g "coarse" -m "svm" -b $B_OPTION -s 1 -o "../output/"


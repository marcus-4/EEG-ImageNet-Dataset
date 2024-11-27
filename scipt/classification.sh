#!/bin/bash

PYTHON_SCRIPT="../src/object_classification.py"

DATA_DIR="../data/"
G_OPTION="all"
# G_OPTION="coarse"
M_OPTION="eegnet"
B_OPTION=80
S_OPTION=0
P_OPTION="eegnet_s${S_OPTION}_1x_0.pth"
O_OPTION="../output/"

#python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -p $P_OPTION -s $S_OPTION -o $O_OPTION
#python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -s $S_OPTION -o $O_OPTION

#My comment out
# for i in {0..15}
# do
#     P_OPTION1="eegnet_s${i}_1x_1.pth"
# #    python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -s $i -o $O_OPTION
#     python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -p $P_OPTION1 -s $i -o $O_OPTION
# done

# python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -p $P_OPTION1 -s $i -o $O_OPTION
python $PYTHON_SCRIPT -d "../data/" -g "coarse" -m "svm" -b $B_OPTION -s 1 -o "../output/"

#!/bin/bash

for pose in $(seq -30 10 30) # pitch
do
        python ./sample_condition.py @@gaze 0,0 @@pose $pose,0
done

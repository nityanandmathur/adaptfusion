#!bin/bash

knockknock slack --webhook-url https://hooks.slack.com/services/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 python ../src/inference.py\
    --source /home/btech/nityanand.mathur/cityscapes/leftImg8bit/source\
    --target /home/btech/nityanand.mathur/adaptfusion/results/000\
    --lora /home/btech/nityanand.mathur/adaptfusion/models/nuimages-samples-sdxl-lora-r4-i1000\
    --layer 000\
    --width 1600\
    --height 900\
    --seed 0

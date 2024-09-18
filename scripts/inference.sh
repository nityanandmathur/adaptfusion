#!bin/bash

knockknock slack --webhook-url https://hooks.slack.com/services/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\
    --channel domain-adaptation\
    python ../src/inference.py\
    --source /home/btech/nityanand.mathur/nuimages/samples\
    --target /home/btech/nityanand.mathur/adaptfusion/results-nu-city/000\
    --lora /home/btech/nityanand.mathur/adaptfusion/models/cityscapes-sdxl-lora-r4-i1000\
    --layer 000\
    --width 1600\
    --height 900\
    --device cuda:1\
    --seed 0

#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

dir=exp/conformer
decode_checkpoint=$dir/final.pt


python wenet/bin/export_onnx.py \
    --config $dir/train.yaml \
    --checkpoint $dir/$decode_checkpoint


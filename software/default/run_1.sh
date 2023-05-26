#!/bin/bash

network=("GCN" "GraphSAGE" "GAT" "GIN" "mix")
for i in `seq 0 19`;
do
    CUDA_VISIBLE_DEVICES=0 python default_main.py ../data/model/test_con_1.txt ${i}
done

#!/bin/bash

network=("GCN" "GraphSAGE" "GAT" "GIN" "mix")
for i in `seq 0 19`;
do
    python default_main.py ../data/model/mix_model.txt ${i}
done

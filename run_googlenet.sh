#!/bin/bash


# googlenet for prediction
python3 googlenet_main.py --target_key=pathology 
python3 googlenet_main.py  --target_key=morph_response 
# python3 googlenet_main.py  --target_key=early_recurrence 
# python3 googlenet_main.py  --target_key=overall_survival_24m 

python3 morph_score_googlenet_main.py 

wait
echo "All jobs finished."
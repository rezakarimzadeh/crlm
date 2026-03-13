#!/bin/bash

set -e

# Multi Instance Learning (MIL) analysis for radiomics features
python3 mil_mlp_main.py --model_name=RadiomicsMIL --target_key=pathology &
python3 mil_mlp_main.py --model_name=RadiomicsMIL --target_key=morph_response &
python3 mil_mlp_main.py --model_name=RadiomicsMIL --target_key=early_recurrence &
python3 mil_mlp_main.py --model_name=RadiomicsMIL --target_key=overall_survival_24m &
python3 morph_score_mil_mlp_main.py --model_name=MorphScoreRadiomicsMIL &

# Statistical pooling analysis for radiomics features
# python3 mil_mlp_main.py --model_name=StatisticalPoolingMLP --target_key=pathology &
# python3 mil_mlp_main.py --model_name=StatisticalPoolingMLP --target_key=morph_response &
# python3 mil_mlp_main.py --model_name=StatisticalPoolingMLP --target_key=early_recurrence &
# python3 mil_mlp_main.py --model_name=StatisticalPoolingMLP --target_key=overall_survival_24m &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreStatisticalPoolingMLP &

wait
echo "All jobs finished."
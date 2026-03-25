#!/bin/bash

set -e

# Multi Instance Learning (MIL) analysis for radiomics features
python3 mil_mlp_main.py --model_name=RadiomicsMIL --target_key=overall_survival_24m > logs/overall_survival_24m_mil.log 2>&1 &
python3 mil_mlp_main.py --model_name=RadiomicsMIL --target_key=pathology > logs/pathology_mil.log 2>&1 &
python3 mil_mlp_main.py --model_name=RadiomicsMIL --target_key=morph_response > logs/morph_response_mil.log 2>&1 &
python3 mil_mlp_main.py --model_name=RadiomicsMIL --target_key=early_recurrence > logs/early_recurrence_mil.log 2>&1 &

# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreRadiomicsMIL --feature_to_include shape boundary intensity texture > logs/morph_score_mil_shape_boundary_intensity_texture.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreRadiomicsMIL --feature_to_include boundary intensity texture > logs/morph_score_mil_boundary_intensity_texture.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreRadiomicsMIL --feature_to_include shape intensity texture > logs/morph_score_mil_shape_intensity_texture.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreRadiomicsMIL --feature_to_include shape boundary texture > logs/morph_score_mil_shape_boundary_texture.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreRadiomicsMIL --feature_to_include shape boundary intensity > logs/morph_score_mil_shape_boundary_intensity.log 2>&1 &

# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreRadiomicsMIL --feature_to_include shape   > logs/morph_score_mil_shape.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreRadiomicsMIL --feature_to_include boundary > logs/morph_score_mil_boundary.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreRadiomicsMIL --feature_to_include intensity > logs/morph_score_mil_intensity.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreRadiomicsMIL --feature_to_include texture > logs/morph_score_mil_texture.log 2>&1 &

# Statistical pooling analysis for radiomics features
#################################################################################################################
python3 mil_mlp_main.py --model_name=StatisticalPoolingMLP --target_key=pathology > logs/pathology_statistical_pooling.log 2>&1 &
python3 mil_mlp_main.py --model_name=StatisticalPoolingMLP --target_key=morph_response > logs/morph_response_statistical_pooling.log 2>&1 &
python3 mil_mlp_main.py --model_name=StatisticalPoolingMLP --target_key=early_recurrence > logs/early_recurrence_statistical_pooling.log 2>&1 &
python3 mil_mlp_main.py --model_name=StatisticalPoolingMLP --target_key=overall_survival_24m > logs/overall_survival_24m_statistical_pooling.log 2>&1 &

# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreStatisticalPoolingMLP --feature_to_include shape boundary intensity texture > logs/morph_score_statistical_pooling_shape_boundary_intensity_texture.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreStatisticalPoolingMLP --feature_to_include boundary intensity texture > logs/morph_score_statistical_pooling_boundary_intensity_texture.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreStatisticalPoolingMLP --feature_to_include shape intensity texture > logs/morph_score_statistical_pooling_shape_intensity_texture.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreStatisticalPoolingMLP --feature_to_include shape boundary texture > logs/morph_score_statistical_pooling_shape_boundary_texture.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreStatisticalPoolingMLP --feature_to_include shape boundary intensity > logs/morph_score_statistical_pooling_shape_boundary_intensity.log 2>&1 &

# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreStatisticalPoolingMLP --feature_to_include shape   > logs/morph_score_statistical_pooling_shape.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreStatisticalPoolingMLP --feature_to_include boundary > logs/morph_score_statistical_pooling_boundary.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreStatisticalPoolingMLP --feature_to_include intensity > logs/morph_score_statistical_pooling_intensity.log 2>&1 &
# python3 morph_score_mil_mlp_main.py --model_name=MorphScoreStatisticalPoolingMLP --feature_to_include texture > logs/morph_score_statistical_pooling_texture.log 2>&1 &





wait
echo "All jobs finished."
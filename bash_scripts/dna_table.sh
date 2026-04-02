#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
RESOLUTION=2

SOURCE_PATHS=(
"data/dna/85_07_table/main/0085_07.smc"
)

    for SOURCE_PATH in "${SOURCE_PATHS[@]}"; do


        OUTPUT_PATH="output_dna_table_r${RESOLUTION}"

        ### FOR THIS SCENE WE USE TIMESTEPS 70-140
       
        # STAGE 1 - train geometry
        python -m scripts.train_stage1 --source_path=$SOURCE_PATH --model_path=$OUTPUT_PATH  \
            --eval --gt_alpha_mask_as_scene_mask --resolution=$RESOLUTION --iterations 35000 \
            --lambda_separation 0.0001 --binarization_warm_up 100 --d_color_reg_loss_weight 0.01 \
            --densify_until_iter 20000 --densification_interval 100 --depth_ratio 1.0 --warm_up 1000 \
            --min_opacity 0.05 --start_normal_reg 6000 --lambda_dist 0 --is_blender --start_frame 70 --end_frame 140

        python -m scripts.render_stage1_insights --source_path=$SOURCE_PATH --model_path=$OUTPUT_PATH  \
            --eval --resolution=$RESOLUTION --load_iter 35000 \
            --depth_ratio 0.0 --load_test_set_only --white_background --is_blender --start_frame 70 --end_frame 140

        # STAGE 2 - train albedo, roughness and envmap

        #We tested x4 and x16 trace_num_rays. x16 slows down optimization but results are slighlu better. You can test different values.
        trace_num_rays=$(((2**18) * 1))

        python -m scripts.train_stage2 --source_path=$SOURCE_PATH --model_path=$OUTPUT_PATH  \
            --eval --gt_alpha_mask_as_scene_mask --resolution=$RESOLUTION --iterations 50000 \
            --load_iter 35000 --diffuse_sample_num 512 --depth_ratio 0.0 --trace_num_rays=$trace_num_rays \
            --envmap_resolution 128 --albedo_lr 0.01 --envmap_cubemap_lr 0.1  --d_lower_hemisphere_weight 0.0  \
            --roughness_lr 0.001 --opacity_lr 0.1 --white_background --is_blender --start_frame 70 --end_frame 140

        # Render materials
        python -m scripts.render_materials --source_path=$SOURCE_PATH --model_path=$OUTPUT_PATH  \
            --eval --resolution=$RESOLUTION --load_iter 50000 \
            --load_test_set_only --white_background --is_blender --start_frame 70 --end_frame 140

        # #render relights

        python -m scripts.render_relight_with_hdr --source_path=$SOURCE_PATH --model_path=$OUTPUT_PATH  \
            --eval --resolution=$RESOLUTION --load_iter 50000 \
            --diffuse_sample_num 2048 --depth_ratio 0.0 --colmap_convention  --load_test_set_only \
            --white_background --is_blender --start_frame 70 --end_frame 140 \
            --hdr example_envmaps/golden_bay_4k_32x16_rot330.hdr

done
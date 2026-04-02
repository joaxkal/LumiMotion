#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
RESOLUTION=2

SOURCE_PATHS=(
"data/enerf_actors_1_3/actor3_895_950"
)

    for SOURCE_PATH in "${SOURCE_PATHS[@]}"; do


        OUTPUT_PATH="output_enerf_actor3_r${RESOLUTION}"

        # STAGE 1 - train geometry
        python -m scripts.train_stage1 --source_path=$SOURCE_PATH --model_path=$OUTPUT_PATH  \
            --eval --gt_alpha_mask_as_scene_mask --resolution=$RESOLUTION --iterations 35000 \
            --lambda_separation 0.0 --d_color_reg_loss_weight 0.001 --densify_until_iter 12000 --densification_interval 300 \
            --depth_ratio 0.0 --warm_up 100 --binarization_warm_up 100 --lambda_alpha_loss 0.002 \
            --min_opacity 0.05 --start_normal_reg 6000 --lambda_dist 0

        python -m scripts.render_stage1_insights --source_path=$SOURCE_PATH --model_path=$OUTPUT_PATH  \
            --eval --resolution=$RESOLUTION --load_iter 35000 \
            --depth_ratio 0.0

        ## STAGE 2 - train albedo, roughness and envmap
        ## Provided scene masks are still quite large - large parts of wall will optimized in Stage2, which may deteriorate material estimation. 
        ## To better focus on shadowed parts, after Stage1 you can remove some wall-gaussians (from areas with no shadow) 
        ## from point_cloud/iteration_35000/point_cloud.ply in SuperSplat and replace the .ply file. We did it for the figures in our paper.

        trace_num_rays=$(((2**18) * 16)) 
        
        python -m scripts.train_stage2 --source_path=$SOURCE_PATH --model_path=$OUTPUT_PATH  \
            --eval --gt_alpha_mask_as_scene_mask --resolution=$RESOLUTION --iterations 45000 \
            --load_iter 35000 --diffuse_sample_num 2048 --depth_ratio 0.0 --trace_num_rays=$trace_num_rays \
            --envmap_resolution 256 --albedo_lr 0.01 --envmap_cubemap_lr 0.1  --d_lower_hemisphere_weight 0.0  \
            --roughness_lr 0.0005 --opacity_lr 0

        # # Render materials
        python -m scripts.render_materials --source_path=$SOURCE_PATH --model_path=$OUTPUT_PATH  \
            --eval --resolution=$RESOLUTION --load_iter 45000

        #render relight
        python -m scripts.render_relight_with_hdr --source_path=$SOURCE_PATH --model_path=$OUTPUT_PATH  \
            --eval --resolution=$RESOLUTION --load_iter 45000 \
            --diffuse_sample_num 2048 --depth_ratio 0.0 --colmap_convention \
            --hdr example_envmaps/golden_bay_4k_32x16_rot330.hdr


done

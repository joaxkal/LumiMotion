#!/usr/bin/env bash
set -euo pipefail

# GPU + base opts
export CUDA_VISIBLE_DEVICES=0
RESOLUTION=2
W_COLOR=0.01
DEPTH_RATIO=0.0

# Map scene nicknames -> source paths
declare -A SCENE_PATHS=(
  [hook]="data/d-nerf-relight-spec32/hook150_v5_spec32"
  [mouse]="data/d-nerf-relight-spec32/mouse150_v5_spec32"
  [jumping jack]="data/d-nerf-relight-spec32/jumpingjacks150_v5_spec32"
  [spheres]="data/d-nerf-relight-spec32/spheres_v5_spec32"
  [standup]="data/d-nerf-relight-spec32/standup150_v5_spec32"
)


# Light combos (train|test|train_name|test_name)
declare -A LIGHTS=(
  [chapelday_goldenbay]="chapel_day_4k_32x16_rot0|golden_bay_4k_32x16_rot330|chapelday|goldenbay"
  [damwall_harbour]="dam_wall_4k_32x16_rot90|small_harbour_sunset_4k_32x16_rot270|damwall|harbour"
  [goldenbay_damwall]="golden_bay_4k_32x16_rot330|dam_wall_4k_32x16_rot90|goldenbay|damwall"
)

# Your configs (combo|scene -> bin weight/lossxyz weight string)
declare -A CONFIGS=(
  ["chapelday_goldenbay|spheres"]="w0.005_lossxyz0.001"
  ["chapelday_goldenbay|jumping jack"]="w0.001_lossxyz0.001"
  ["chapelday_goldenbay|hook"]="w0.001_lossxyz0.001"
  ["chapelday_goldenbay|standup"]="w0.005_lossxyz0.0"
  ["chapelday_goldenbay|mouse"]="w0.005_lossxyz0.001"

  ["damwall_harbour|spheres"]="w0.005_lossxyz0.0"
  ["damwall_harbour|jumping jack"]="w0.001_lossxyz0.0"
  ["damwall_harbour|hook"]="w0.001_lossxyz0.001"
  ["damwall_harbour|standup"]="w0.001_lossxyz0.001"
  ["damwall_harbour|mouse"]="w0.001_lossxyz0.0"

  ["goldenbay_damwall|spheres"]="w0.005_lossxyz0.001"
  ["goldenbay_damwall|jumping jack"]="w0.001_lossxyz0.0"
  ["goldenbay_damwall|hook"]="w0.001_lossxyz0.001"
  ["goldenbay_damwall|standup"]="w0.001_lossxyz0.001"
  ["goldenbay_damwall|mouse"]="w0.005_lossxyz0.001"
)

OUTPUT_BASE="outputs_sanity_moresep"

for KEY in "${!CONFIGS[@]}"; do
    COMBO="${KEY%%|*}"
    SCENE="${KEY#*|}"
    CFG="${CONFIGS["$KEY"]}"

    # Parse wbin / wxyz from strings like: w0.005_lossxyz0.001
    W_BIN=$(echo "$CFG"    | sed -E 's/.*_w([0-9.]+)_lossxyz.*/\1/')
    W_XYZ=$(echo "$CFG"    | sed -E 's/.*lossxyz([0-9.]+).*/\1/')

    IFS='|' read -r TRAIN_LIGHT TEST_LIGHT TRAIN_NAME TEST_NAME <<< "${LIGHTS[$COMBO]}"

    SOURCE_PATH="${SCENE_PATHS["$SCENE"]}"
    SCENE_NAME="$(basename "$SOURCE_PATH")"
    OUTPUT_PATH="${OUTPUT_BASE}/${TRAIN_NAME}_${TEST_NAME}/${SCENE_NAME}_r${RESOLUTION}"


    echo "--------------------------------------------------------------------------------"
    echo "Processing: $SCENE  ($SCENE_NAME)"
    echo " Source: $SOURCE_PATH"
    echo " Lights: Train=$TRAIN_LIGHT ($TRAIN_NAME) | Test=$TEST_LIGHT ($TEST_NAME)"
    echo " Params: start=$START_BIN, wbin=$W_BIN, wxyz=$W_XYZ, wcolor=$W_COLOR, depth=$DEPTH_RATIO"
    echo " Output: $OUTPUT_PATH"
    echo "--------------------------------------------------------------------------------"

    # ===== STAGE 1 - train geometry =====
     python -m scripts.train_stage1 --source_path="$SOURCE_PATH" --model_path="$OUTPUT_PATH" \
       --is_blender --eval --gt_alpha_mask_as_scene_mask --resolution="$RESOLUTION" --iterations 35000 \
       --train_light_folder "$TRAIN_LIGHT" --densify_until_iter 20000  \
       --lambda_separation "$W_BIN" --d_xyz_loss_weight "$W_XYZ" --binarization_warm_up 1000 \
       --depth_ratio 1.0 --d_color_reg_loss_weight "$W_COLOR" # keep depth ratio to 1.0 for stage1 for synth scenes!

     python -m scripts.render_stage1_insights --source_path="$SOURCE_PATH" --model_path="$OUTPUT_PATH"  \
             --is_blender --eval --resolution=$RESOLUTION --load_iter 35000 \
             --train_light_folder $TRAIN_LIGHT --depth_ratio "$DEPTH_RATIO"

    # # ===== STAGE 2 - train albedo, roughness and envmap =====
    python -m scripts.train_stage2 --source_path="$SOURCE_PATH" --model_path="$OUTPUT_PATH"  \
      --is_blender --eval --gt_alpha_mask_as_scene_mask --resolution="$RESOLUTION" --iterations 55000 \
      --load_iter 35000 --diffuse_sample_num 512  \
      --train_light_folder "$TRAIN_LIGHT" --depth_ratio "$DEPTH_RATIO"

    # ===== Render materials =====
    python -m scripts.render_materials --source_path="$SOURCE_PATH" --model_path="$OUTPUT_PATH"  \
      --is_blender --eval --resolution="$RESOLUTION" --load_iter 55000  \
      --train_light_folder "$TRAIN_LIGHT" --depth_ratio "$DEPTH_RATIO"

    # ===== Static eval, specify test_light_folder =====
    python -m scripts.scale_albedo_static \
      --source_path "$SOURCE_PATH" --model_path "$OUTPUT_PATH" \
      --eval --is_blender --load_iter 55000  \
      --train_light_folder "$TRAIN_LIGHT" --test_light_folder "$TEST_LIGHT" --resolution "$RESOLUTION"  --depth_ratio "$DEPTH_RATIO"

    python -m scripts.eval_material_static \
      --source_path "$SOURCE_PATH" --model_path "$OUTPUT_PATH" \
      --eval --is_blender --load_iter 55000  \
      --train_light_folder "$TRAIN_LIGHT" --test_light_folder "$TEST_LIGHT" --resolution "$RESOLUTION"  --depth_ratio "$DEPTH_RATIO"

    python -m scripts.eval_relight_static \
      --source_path "$SOURCE_PATH" --model_path "$OUTPUT_PATH" \
      --eval --is_blender --load_iter 55000  \
      --train_light_folder "$TRAIN_LIGHT" --test_light_folder "$TEST_LIGHT" --resolution "$RESOLUTION" \
      --diffuse_sample_num 2048  --depth_ratio "$DEPTH_RATIO"

    # IMPORTANT! FOR NVS if any opt params for training were specified (like envmap activation or envmap resolution) you NEED to pass them here as well
    python -m scripts.eval_nvs_static \
      --source_path "$SOURCE_PATH" --model_path "$OUTPUT_PATH" \
      --eval --is_blender --load_iter 55000 \
      --train_light_folder "$TRAIN_LIGHT" --resolution "$RESOLUTION" \
      --diffuse_sample_num 2048  --depth_ratio "$DEPTH_RATIO"

    # ===== Dynamic eval, specify test_light_folder =====
    python -m scripts.scale_albedo_dynamic \
      --source_path "$SOURCE_PATH" --model_path "$OUTPUT_PATH" \
      --eval --is_blender --load_iter 55000   \
      --train_light_folder "$TRAIN_LIGHT" --test_light_folder "$TEST_LIGHT" --resolution "$RESOLUTION"  --depth_ratio "$DEPTH_RATIO"

    python -m scripts.eval_material_dynamic \
      --source_path "$SOURCE_PATH" --model_path "$OUTPUT_PATH" \
      --eval --is_blender --load_iter 55000  \
      --train_light_folder "$TRAIN_LIGHT" --test_light_folder "$TEST_LIGHT" --resolution "$RESOLUTION"  --depth_ratio "$DEPTH_RATIO"

    python -m scripts.eval_relight_dynamic \
      --source_path "$SOURCE_PATH" --model_path "$OUTPUT_PATH" \
      --eval --is_blender --load_iter 55000   \
      --train_light_folder "$TRAIN_LIGHT" --test_light_folder "$TEST_LIGHT" --resolution "$RESOLUTION" \
      --diffuse_sample_num 2048  --depth_ratio "$DEPTH_RATIO"

    # IMPORTANT! FOR NVS if any opt params for training were specified (like envmap activation or envmap resolution) you NEED to pass them here as well
    python -m scripts.eval_nvs_dynamic \
      --source_path "$SOURCE_PATH" --model_path "$OUTPUT_PATH" \
      --eval --is_blender --load_iter 55000 \
      --train_light_folder "$TRAIN_LIGHT" --resolution "$RESOLUTION" \
      --diffuse_sample_num 2048  --depth_ratio "$DEPTH_RATIO"

    # ===== Extra render =====
    # python -m scripts.render_relight_with_rotating_envmap \
    #   --source_path "$SOURCE_PATH" --model_path "$OUTPUT_PATH" \
    #   --eval --is_blender --load_iter 55000   \
    #   --train_light_folder "$TRAIN_LIGHT" --resolution "$RESOLUTION" \
    #   --diffuse_sample_num 2048  --depth_ratio "$DEPTH_RATIO"

done
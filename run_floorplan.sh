#!/bin/bash

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

# Use system python
PYTHON="python3"

# Path to Model #1 (CGAN constraint-based generator)
GEN_SCRIPT="./generate_floorplan.py"

# Where pix2pixHD expects input
TEST_A="./datasets/floorplans/test_A"

# Ensure directories exist
mkdir -p "$TEST_A"
mkdir -p "./final_output"

echo "-------------------------------------------------"
echo "▶ Running CGAN generator (Model #1)…"
echo "-------------------------------------------------"

$PYTHON $GEN_SCRIPT
if [ ! -f "./generated_floorplan.png" ]; then
  echo "❌ ERROR: CGAN did not generate image (generated_floorplan.png missing)"
  exit 1
fi

# Move the image to pix2pixHD dataset folder
TS=$(date +%Y%m%d_%H%M%S)
DEST="$TEST_A/input_$TS.png"
mv "./generated_floorplan.png" "$DEST"

echo "✅ Rough layout generated and placed here:"
echo "   → $DEST"


echo
echo "-------------------------------------------------"
echo "🚀 Running Pix2PixHD refinement (Model #2)…"
echo "-------------------------------------------------"

$PYTHON ./pix2pixHD/test.py \
  --name floorGAN_finetune_v3 \
  --checkpoints_dir ./pix2pixHD/checkpoints \
  --results_dir ./pix2pixHD/results \
  --dataroot ./datasets/floorplans \
  --label_nc 0 \
  --no_instance \
  --netG global \
  --n_downsample_global 3 \
  --n_blocks_global 4 \
  --resize_or_crop none \
  --gpu_ids 0 \
  --which_epoch latest \
  --how_many 1


# Find refined output from pix2pixHD
OUT_IMG=$(ls ./pix2pixHD/results/floorGAN_finetune_v3/test_latest/images/*_synthesized_image.jpg 2>/dev/null | tail -n 1)

if [ -z "$OUT_IMG" ]; then
  echo "⚠️ ERROR: Pix2PixHD didn't output any refined image."
  exit 1
fi

cp "$OUT_IMG" "./final_output/final_$TS.jpg"

echo "✅ Final refined output saved at:"
echo "   → ./final_output/final_$TS.jpg"
echo
echo "🎉 Pipeline Completed Successfully"


export CUDA_VISIBLE_DEVICES=0,1;

for value in 1 2 4 8 16 32 64 80; do
    accelerate launch --num_processes 2 scripts/coco_detection.py \
        ++coco.classes_per_call=$value ++model.path="/path/to/models/paligemma2-10b-pt-896" ++generation.batch_size=20
done
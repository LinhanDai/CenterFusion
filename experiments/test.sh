export CUDA_VISIBLE_DEVICES=0
cd src

## Perform detection and evaluation
python3 test.py ddd \
    --exp_id centerfusion \
    --dataset nuscenes \
    --val_split mini_val \
    --run_dataset_eval \
    --num_workers 4 \
    --nuscenes_att \
    --velocity \
    --gpus 0 \
    --pointcloud \
    --radar_sweeps 1 \
    --max_pc_dist 60.0 \
    --pc_z_offset -0.0 \
    --load_model ../exp/ddd/centerfusion/model_last.pth \
    --flip_test \
    --debug 4 \
    # --resume \

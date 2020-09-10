CUDA_VISIBLE_DEVICES=0,1,2,3 python trainer_aaai.py --data-root /home/ziheng/datasets/gaze \
--batch-size-train 64 --batch-size-val 128 --num-workers 0 --exp-name gaze_aaai_refine_headpose \
 - train_headpose --epochs 10 --lr 1e-2\
 - train_headpose --epochs 15 --lr 1e-3\
 - train_headpose --epochs 17 --lr 1e-4\
 - end


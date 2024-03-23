#export NCCL_IB_DISABLE=1
python3 testing.py  --num_gpus 1                       \
 --coco_path /tmp/pycharm_project_DETR                  \
 --num_queries 1000  --two_stage --resume checkpoint-52_two_stage.pth \
 --batch_size 1 --start_epoch 49 --end_epoch 55    \
 --with_box_refine  --aps 0 --AMself 0 --DeA 0 --ARelation 0 --output_dir two-stage-deformable
python3 demo_opt.py record-final 49

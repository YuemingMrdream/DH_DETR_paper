#export NCCL_IB_DISABLE=1
#bash ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh   \
#   --num_queries 1000 --epochs 54 --enc_layers 6 --dec_layers 6 \
#   --with_box_refine  --lr_drop 40 --batch_size 2 --aps 0 --AMself 0 --two_stage  \
#   --DeA 0 --ARelation 0 --output_dir two-stage-deformable-6-6

python3 testing.py  --num_gpus 2                       \
 --coco_path /tmp/pycharm_project_DETR                  \
 --num_queries 500  --two_stage                        \
 --batch_size 1 --start_epoch 49 --end_epoch 50 --enc_layers 6 --dec_layers 6   \
 --with_box_refine  --aps 0 --AMself 0 --DeA 0 --ARelation 0 --output_dir two-stage-deformable-6-6-500
python3 demo_opt.py record-final 40



#
#export NCCL_IB_DISABLE=1
#bash ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh   \
#   --num_queries 500 --epochs 54 --enc_layers 6 --dec_layers 3 \
#   --with_box_refine  --lr_drop 40 --batch_size 2 --aps 1 --AMself 0 --two_stage  \
#   --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-3-1-ONLY-ID_token-subtract-norm-addq-Giou500
#
#python3 testing.py  --num_gpus 4                       \
# --coco_path /tmp/pycharm_project_DETR                  \
# --num_queries 500  --two_stage                        \
# --batch_size 1 --start_epoch 40 --end_epoch 55 --enc_layers 6 --dec_layers 3    \
# --with_box_refine  --aps 1 --AMself 0 --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-3-1-ONLY-ID_token-subtract-norm-addq-Giou500
#python3 demo_opt.py record-final 40
#
#


#export NCCL_IB_DISABLE=1
#bash ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh   \
#   --num_queries 500 --epochs 54 --enc_layers 6 --dec_layers 4 \
#   --with_box_refine  --lr_drop 40 --batch_size 2 --aps 1 --AMself 0 --two_stage  \
#   --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-4-1-ONLY-ID_token-subtract-norm-addq-Giou500 --resume /root/autodl-tmp/Iter-deformable-AHA/DH_head/model_dump/two-stage-deformable-ADis-real-4-1-ONLY-ID_token-subtract-norm-addq--Giou500/checkpoint.pth
#
#python3 testing.py  --num_gpus 4                       \
# --coco_path /tmp/pycharm_project_DETR                  \
# --num_queries 500  --two_stage                        \
# --batch_size 1 --start_epoch 40 --end_epoch 55 --enc_layers 6 --dec_layers 4    \
# --with_box_refine  --aps 1 --AMself 0 --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-4-1-ONLY-ID_token-subtract-norm-addq-Giou500
#python3 demo_opt.py record-final 40
#
#export NCCL_IB_DISABLE=1
#bash ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh   \
#   --num_queries 500 --epochs 54 --enc_layers 6 --dec_layers 4 \
#   --with_box_refine  --lr_drop 40 --batch_size 2 --aps 2 --AMself 0 --two_stage  \
#   --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-4-2-ONLY-ID_token-subtract-norm-addq-Giou500
#
#python3 testing.py  --num_gpus 4                       \
# --coco_path /tmp/pycharm_project_DETR                  \
# --num_queries 500  --two_stage                        \
# --batch_size 1 --start_epoch 40 --end_epoch 55 --enc_layers 6 --dec_layers 4    \
# --with_box_refine  --aps 2 --AMself 0 --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-4-2-ONLY-ID_token-subtract-norm-addq-Giou500
#python3 demo_opt.py record-final 40
#
#
#bash ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh   \
#   --num_queries 500 --epochs 54 --enc_layers 6 --dec_layers 4 \
#   --with_box_refine  --lr_drop 40 --batch_size 2 --aps 1 --AMself 0 --two_stage  \
#   --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-4-1-ONLY-ID_token-subtract-norm-addq-Giou500 --resume /root/autodl-tmp/Iter-deformable-AHA/DH_head/model_dump/two-stage-deformable-ADis-real-4-1-ONLY-ID_token-subtract-norm-addq--Giou500/checkpoint.pth
#
#python3 testing.py  --num_gpus 4                       \
# --coco_path /tmp/pycharm_project_DETR                  \
# --num_queries 500  --two_stage                        \
# --batch_size 1 --start_epoch 40 --end_epoch 55 --enc_layers 6 --dec_layers 4    \
# --with_box_refine  --aps 1 --AMself 0 --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-4-1-ONLY-ID_token-subtract-norm-addq-Giou500
#python3 demo_opt.py record-final 40
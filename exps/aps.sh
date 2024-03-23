export NCCL_IB_DISABLE=1
bash ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh   \
   --coco_path  /root/autodl-tmp/Crowdhuman --num_queries 500 --epochs 54 --enc_layers 6 --dec_layers 3 \
   --with_box_refine  --lr_drop 40 --batch_size 2 --aps 2  --two_stage  \
  --ARelation 1 --output_dir two-stage-deformable-ADis-real-3-2-ONLY-ID_token-subtract-norm-addq-Giou500-TRY


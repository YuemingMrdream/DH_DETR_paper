export NCCL_IB_DISABLE=1
python3 testing.py  --num_gpus 1                      \
 --coco_path  /root/autodl-tmp/Crowdhuman --num_queries 500  --two_stage  \
 --batch_size 1 --start_epoch 40 --end_epoch 55 --enc_layers 6 --dec_layers 3    \
 --with_box_refine  --aps 2  --ARelation 1 --output_dir two-stage-deformable-ADis-real-3-2-ONLY-ID_token-subtract-norm-addq-Giou500-TRY

# testing AP,MR,JI
python3 demo_opt.py final 40


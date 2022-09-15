nohup python3 -u run.py \
--exp_name BERT_QA_Lightning \
--squad_v2 False \
--per_gpu_batch_size 16 \
--val_check_interval 100 \
--accumulate_grad_batches 4 \
--limit_val_batches 30 \
--do_train True \
> train.log 2>&1 &
#> train.log 2>&1 &
#--do_train True \
#--args.default_root_dir \

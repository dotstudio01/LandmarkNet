python3 tools/convert_checkpoint.py $2/epoch_049.pth.tar
python3 -u test.py  \
--batch_size 6  \
--seq_len 2  \
--num_workers 10  \
--cuda 1  \
--checkpoint 'checkpoints/attention_loc.pth.tar'  \
--logdir $2  \
--dataset 'CambridgeLandmarks'  \
--data_dir '/storage/dataset/CambridgeLandmarks'  \
--scene $1 \
--split 'test' \
--normalize_pose 0  \
--seed 7
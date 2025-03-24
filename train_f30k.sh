#train
time=`date '+%Y-%m-%d-%H-%M'`
path='output/'$time
data_path='/home/lzy/data/f30k'
vocab_path='/home/lzy/data/vocab'
gpu=3
module_name=SGR

python3 -u run.py \
    --id train_f30k \
    --gpu $gpu \
    --workers 2 \
    --seed 96 \
    --warmup_epoch 10 \
    --warmup_type warmup_sele \
    --data_name f30k_precomp \
    --noise_ratio 0.1 \
    --warmup_rate 0.3 \
    --p_threshold 0.5 \
    --noise_train noise_soft \
    --noise_tem 0.5 \
    --fit_type bmm \
    --noise_file noise_index/f30k_precomp_0.4.npy \
    --module_name $module_name \
    --output_dir 'output/'$time\
    --data_path $data_path \
    --vocab_path $vocab_path
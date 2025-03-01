#!/bin/bash

gpus=2
node_num=1
single_gpus=`expr $gpus / $node_num`

cpus=4

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

while true
do 
    PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))
    break
done
echo $PORT




# reserved
srun -p ai4earth --kill-on-bad-exit=1 --job-name=lgunet_vae_large -x SH-IDC1-10-140-24-17 --quotatype=spot --ntasks-per-node=$single_gpus --cpus-per-task=10 -N $node_num -o job2/%j.out --gres=gpu:$single_gpus --async python -u train.py \
--init_method 'tcp://127.0.0.1:'$PORT   \
-c ./config/fengwu.yaml \
--world_size $gpus   \
--per_cpus $cpus    \
--tensor_model_parallel_size 1                  \
--pipeline_model_parallel_size 1                  \
--outdir './test_results' \
--resume_from_config                        \
--desc   'fengwu'    


sleep 2
rm -f batchscript-*


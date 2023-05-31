# !/usr/bin/env bash
GPUS=4
batchsize=4

SUBDIR=PATrans

CONFIG=asuvid_R_50_${SUBDIR}
#CONFIG=cvcvid_R_50_${SUBDIR}

save_path=log_dir/${SUBDIR}/${CONFIG}

if [ ! -d ${save_path} ]; then
  mkdir -p ${save_path}
  cp configs/${SUBDIR}/${CONFIG}.yaml ${save_path}/
fi


srun --mpi=pmi2 -p <partition> -n${GPUS} --gres=gpu:${GPUS} --ntasks-per-node=${GPUS} \
  --cpus-per-task=4 \
  --job-name=${save_path} \
  python -u tools/train_net.py \
  --master_port=$((RANDOM + 11111)) \
  --config-file ${save_path}/${CONFIG}.yaml \
  OUTPUT_DIR ${save_path} \
  MODEL.WEIGHT pretrained_models/R-50.pkl
  2>&1|tee ${save_path}/train.log &



#!/usr/bin/env bash
GPUS=4
batchsize=4

model_name=  # path of trained model (.pkl, .pth)

SUBDIR=PATrans
#CONFIG=asuvid_R_50_PATrans
CONFIG=cvcvid_R_50_PATrans
save_path=log_dir/${CONFIG}/inference_result
if [ ! -d ${save_path} ]; then
  mkdir -p ${save_path}
fi

srun --mpi=pmi2 -p <partition> -n${GPUS} --gres=gpu:${GPUS} --ntasks-per-node=${GPUS} \
  --cpus-per-task=4 \
  --job-name=inference_${save_path} \
  python -u tools/test_net.py \
  --master_port=$((RANDOM + 10000)) \
  --config-file configs/${SUBDIR}/${CONFIG}.yaml \
  --visulize \
  MODEL.WEIGHT ${model_name} \
  OUTPUT_DIR ${save_path} \
  TEST.IMS_PER_BATCH ${batchsize} \
  2>&1|tee ${save_path}/inference.log &


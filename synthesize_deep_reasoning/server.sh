
set -ex
export HOST_IP=0.0.0.0
cname=/path/to/config.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server --model $model --port $port --disable-log-requests --max-model-len 32000 -tp 4 --gpu-memory-utilization 0.6 --trust-remote-code &  
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server --model $model2 --port 2$port --disable-log-requests --max-model-len 32000 -tp 2 --gpu-memory-utilization 0.6 --trust-remote-code &

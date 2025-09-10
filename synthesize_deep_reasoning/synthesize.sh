set -ex
export HOST_IP=0.0.0.0
model2=${model2:-""}
cd ${workdir}
python synthesize.py --port $port --rank ${rank} --total-ranks ${total} --config_file $cname --model $model --posterior_model $model2 # port rank total

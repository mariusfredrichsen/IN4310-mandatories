module load PyTorch-bundle/1.10.0-MKL-bundle-pre-optimised

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=5 nohup python3 main.py > out1.log 2> error1.log &

nvidia-smi

ps -u <user-name> | grep -i python

kill -9 <process-id>

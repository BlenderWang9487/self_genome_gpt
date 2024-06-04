set -ex

WKEY=$1
HKEY=$2


#### all pip install  (might have some issues with conda / pip conflict)
pip install lightning transformers datasets packaging typer[all] wandb

# Flash attention 2 is installed separately,
# (this is optional, you can use torch builtin sdpa and skip this)
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation

wandb login $WKEY
huggingface-cli login --token $HKEY
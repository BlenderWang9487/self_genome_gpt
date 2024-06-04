# SelfGenomeGPT

Inspired by Self-GenomeNet

> Gündüz, H.A., Binder, M., To, XY. et al. A self-supervised deep learning method for data-efficient training in genomics. Commun Biol 6, 928 (2023). https://doi.org/10.1038/s42003-023-05310-2


I try to use transformer decoder, instead of RNN in original paper, to do the similar sequence contrastive learning with Reverse-Complement neighbor

## Usage

```python
import torch

from sggpt import SelfGenomeGPT, SelfGenomeGPTConfig

config = SelfGenomeGPTConfig(attn_impl="sdpa")
model = SelfGenomeGPT(config)

x = torch.randint(0, 4, (1, config.max_len))
out = model.forward(x)
print(out.hidden_states)
```
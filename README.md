# SelfGenomeGPT

inspired by Self-GenomeNet
Transformer sequence contrastive learning with Reverse-Complement neighbor

# model upload 

```bash
huggingface-cli upload \
    <repo> \
    <model path> \
    <remote path, usaully "./"> \
    --repo-type model \
    --private 
```

# model download 

```bash
huggingface-cli download \
    <repo> \
    --repo-type model \
    --local-dir <output dir> \
    --local-dir-use-symlinks False
```
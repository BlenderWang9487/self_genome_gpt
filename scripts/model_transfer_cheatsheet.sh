# model upload 

huggingface-cli upload \
    <repo> \
    <model path> \
    <remote path, usaully "./"> \
    --repo-type model \
    --private 


# model download 

huggingface-cli download \
    <repo> \
    --repo-type model \
    --local-dir <output dir> \
    --local-dir-use-symlinks False
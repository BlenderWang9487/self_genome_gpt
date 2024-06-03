import numpy as np
import torch

from .contants import NUC_MAP, NUM_NUCS


def rev_comp(int_seq: np.ndarray) -> np.ndarray:
    return np.flip((NUM_NUCS - 1) - int_seq, axis=0)


class PretrainCollator:
    def __init__(self):
        self.str2int = np.vectorize(lambda x: NUC_MAP[x] if x in NUC_MAP else 0)

    def __call__(self, example: dict):
        seq = example["sequence"]
        seq = self.str2int(list(seq))
        return seq

    def batch(self, examples: list[dict]):
        seqs = [self(e) for e in examples]
        rev_seqs = [rev_comp(s) for s in seqs]

        np_batch = np.stack(seqs + rev_seqs, axis=0, dtype=np.int64)
        return torch.from_numpy(np_batch)

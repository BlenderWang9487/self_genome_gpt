NUC_MAP = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 0,  # N is treated as A for convenience
}  # so rev comp will be reverse(3 - s)
NUC_LIST = list(NUC_MAP.keys())
NUM_NUCS = 4  # A, C, G, T, just ignore N

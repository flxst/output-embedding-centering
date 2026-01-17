
SINGLE_CHECKPOINT = False

BATCH_SIZE = 48

MICRO_BATCH_SIZE = {
    # note that these values are optimized for 40GB RAM

    # nanogpt
    "125M": 12,
    "355M": 6,
    "760M": 6,
    "1300M": 3,

    # porian
    # 1 GPU
    '5M': 48,   # 1 -> bs = mbs = 20
    '8M': 48,   # 2 -> bs = mbs = 28
    '10M': 48,  # 3 -> bs = mbs = 32
    '16M': 48,  # 4 -> bs = mbs = 44

    # 2 GPUs
    '23M': 48,  # 5 -> bs = 56, mbs = 28
    '29M': 48,  # 6 -> bs = 64, mbs = 32

    # 4 GPUs
    '37M': 48,  # 7 -> bs = 80, mbs = 20
    '57M': 48,  # 8 -> bs = 104, mbs = 26
    '85M': 24,  # 9 -> bs = 128, mbs = 16
    '109M': 24, # A -> bs = 160, mbs = 20
    '149M': 20, # B -> bs = 192, mbs = 16
    # Porian et al. use beta_2 = 0.99 above this line and 0.95 below
    '221M': 16,  # C -> bs = 256, mbs = 8  
    '347M': 12,  # D -> bs = 320, mbs = 10
    '455M': 8,  # E -> bs = 448, mbs = 4
    '612M': 6,  # F -> bs = 512, mbs = 4
    '902M': 4,  # G -> bs = 640, mbs = 2
}

BLOCK_SIZE = 2048

GPUS = {
    "node": 4,

    # 1 GPU
    '5M': 4,
    '8M': 4,
    '10M': 4,
    '16M': 4,

    # 2 GPUs
    '23M': 4,
    '29M': 4,

    # 4 GPUs
    '37M': 4,
    '57M': 4,
    '85M': 4,
    '109M': 4,
    '149M': 4,
    '221M': 4,
    '347M': 4,
    '455M': 4,
    '612M': 4,
    '902M': 4,
}

LEARNING_RATE = {
    "125M": '3e-4',
    "355M": '3e-4',
    "760M": '2.5e-4',
    "1300M": '2.0e-4',
}

WARMUP = 100
WARMUP_WORTSMAN = 5000

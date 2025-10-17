from torch.optim import AdamW


class AdamATan2(AdamW):
    """Compatibility shim for environments without a working adam-atan2 CUDA kernel.

    Uses standard AdamW with the same constructor signature.
    """
    pass


import torch

def length_to_mask(length, max_len=None, dtype=None):
    """length: [B].
    return [B x max_len].
    If max_len is None, then max of length will be used.
    """
    assert len(length.size()) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

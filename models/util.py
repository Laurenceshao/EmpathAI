import torch


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda')
        # elif torch.backends.mps.is_available():
        #     return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')
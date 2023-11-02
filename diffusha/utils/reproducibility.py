import torch


def set_deterministic():
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    print("torch runs deteministically!!")

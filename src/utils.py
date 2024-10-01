import torch 

def get_torch_device():
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        return "cuda"
    else:
        return "mps"
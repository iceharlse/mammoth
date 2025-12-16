import torch

state = torch.load("alpha_seqcifar10_V3.pth", map_location="cpu")
print(type(state))       # 应该是 dict
print(list(state.keys())[:5])

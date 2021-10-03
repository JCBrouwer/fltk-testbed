import torch

from . import *

print("compiling AnyCostGenerator...")
AnyCostGenerator(256).cuda().forward(torch.randn(4, 512, device="cuda"), None)
print("compiling MobileStyleGenerator...")
MobileStyleGenerator(256).cuda().forward(torch.randn(4, 512, device="cuda"), None)
print("compiling Style1Generator...")
Style1Generator(256).cuda().forward(torch.randn(4, 512, device="cuda"), None)
print("compiling Style2Generator...")
Style2Generator(256).cuda().forward(torch.randn(4, 512, device="cuda"), None)
print("compiling StyleMapGenerator...")
StyleMapGenerator(256).cuda().forward(torch.randn(4, 512, device="cuda"), None)
print("compiling SWAGenerator...")
SWAGenerator(256).cuda().forward(torch.randn(4, 512, device="cuda"), None)
print("compiling Style2ADAGenerator...")
Style2ADAGenerator(256).cuda().forward(torch.randn(4, 512, device="cuda"), None)

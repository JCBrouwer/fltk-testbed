import os
import re
from time import time

# monkey patch all the custom CUDA ops to build into their own separate directory to avoid having to rebuild EVERY time
for filename in [
    "fltk/nets/gan/StyleMapGAN/training/op/fused_act.py",
    "fltk/nets/gan/StyleMapGAN/training/op/upfirdn2d.py",
    "fltk/nets/gan/MobileStyleGAN.pytorch/core/models/modules/ops/fused_act_cuda.py",
    "fltk/nets/gan/MobileStyleGAN.pytorch/core/models/modules/ops/upfirdn2d_cuda.py",
    "fltk/nets/gan/anycost-gan/cuda_op/fused_act.py",
    "fltk/nets/gan/anycost-gan/cuda_op/upfirdn2d.py",
    "fltk/nets/gan/stylegan2-pytorch/op/conv2d_gradfix.py",
    "fltk/nets/gan/stylegan2-pytorch/op/fused_act.py",
    "fltk/nets/gan/stylegan2-pytorch/op/upfirdn2d.py",
]:
    with open(filename, "r") as file:
        filedata = file.read()
    build_dir = os.path.abspath("/".join(file.name.split("/")[:4])) + "/opbuild/"
    os.makedirs(build_dir, exist_ok=True)
    filedata = re.sub(
        r"(load\(\n.*\n.*\n.*\n.*\n.*],)\n\)",
        r'\1\n    build_directory="' + build_dir + '",\n)',
        filedata,
    )
    with open(filename, "w") as file:
        file.write(filedata)

t = time()
print("importing")
from .anycost import AnyCostGenerator
from .mobile import MobileStyleGenerator
from .style1 import Style1Generator
from .style2 import Style2Generator
from .stylemap import StyleMapGenerator
from .swa import SWAGenerator

print(time() - t)

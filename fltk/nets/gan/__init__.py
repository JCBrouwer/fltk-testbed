import os
import re
import shutil

# monkey patch all the custom CUDA ops to build into their own separate directory to avoid having to rebuild EVERY time
shutil.copytree("fltk/nets/gan/stylegan2-pytorch/op", "fltk/nets/gan/anycost-gan/cuda_op", dirs_exist_ok=True)
shutil.copytree("fltk/nets/gan/stylegan2-pytorch/op", "fltk/nets/gan/StyleMapGAN/training/op", dirs_exist_ok=True)
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
    "fltk/nets/gan/stylegan2-ada-pytorch/torch_utils/ops/bias_act.py",
    "fltk/nets/gan/stylegan2-ada-pytorch/torch_utils/ops/conv2d_gradfix.py",
    "fltk/nets/gan/stylegan2-ada-pytorch/torch_utils/ops/conv2d_resample.py",
    "fltk/nets/gan/stylegan2-ada-pytorch/torch_utils/ops/fma.py",
    "fltk/nets/gan/stylegan2-ada-pytorch/torch_utils/ops/grid_sample_gradfix.py",
    "fltk/nets/gan/stylegan2-ada-pytorch/torch_utils/ops/upfirdn2d.py",
    "fltk/nets/gan/stylegan2-ada-pytorch/torch_utils/custom_ops.py",
]:
    with open(filename, "r") as file:
        filedata = file.read()
    build_dir = f"fltk/ops/{filename.split('/')[3]}/"
    os.makedirs(build_dir, exist_ok=True)
    filedata = re.sub(
        r"(load\(\n.*\n.*\n.*\n.*\n.*],)\n\)",
        r'\1\n    build_directory="' + build_dir + '",\n)',
        filedata,
    ).replace("fltk/ops/stylegan2-pytorch/", build_dir)
    filedata = filedata.replace(
        'if any(torch.__version__.startswith(x) for x in ["1.7.", "1.8."]):',
        'if any(torch.__version__.startswith(x) for x in ["1.7.", "1.8.", "1.9.", "1.10.", "1.11."]):',
    ).replace(
        "if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9']):",
        'if any(torch.__version__.startswith(x) for x in ["1.7.", "1.8.", "1.9.", "1.10.", "1.11."]):',
    )
    filedata = filedata.replace(
        "verbosity = 'brief'",
        "verbosity = 'none'",
    )
    filedata = filedata.replace(
        "enabled = False",
        "enabled = True",
    )
    with open(filename, "w") as file:
        file.write(filedata)

import torch

torch.backends.cudnn.benchmark = True

from .anycost import AnyCostGenerator
from .mobile import MobileStyleGenerator
from .style1 import Style1Generator
from .style2 import Style2Generator
from .style2ada import Style2ADAGenerator
from .stylemap import StyleMapGenerator
from .swa import SWAGenerator

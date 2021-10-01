import torch
import torch.nn.functional as F


def gaussian_filter(x, sigma, causal=None, mode="circular"):
    dim = len(x.shape)
    n_frames = x.shape[0]
    while len(x.shape) < 3:
        x = x[:, None]

    radius = min(int(sigma * 4), 3 * len(x))
    channels = x.shape[1]

    kernel = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-0.5 / sigma ** 2 * kernel ** 2)
    if causal is not None:
        kernel[radius + 1 :] *= 0 if not isinstance(causal, float) else causal
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    if radius > n_frames:  # prevent padding errors on short sequences
        x = F.pad(x, (n_frames, n_frames), mode=mode)
        print(
            f"WARNING: Gaussian filter radius ({int(sigma * 4 )}) is larger than number of frames ({n_frames}).\n\t Filter size has been lowered to ({radius}). You might want to consider lowering sigma ({sigma})."
        )
        x = F.pad(x, (radius - n_frames, radius - n_frames), mode="replicate")
    else:
        x = F.pad(x, (radius, radius), mode=mode)

    x = F.conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)

    if len(x.shape) > dim:
        x = x.squeeze()

    return x

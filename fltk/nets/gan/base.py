import torch


class InferenceGenerator(torch.nn.Module):
    def __init__(self, module, size):
        super().__init__()
        self.module = module(size)

    def forward(self, latent, noise):
        return self.module(latent, noise=noise)
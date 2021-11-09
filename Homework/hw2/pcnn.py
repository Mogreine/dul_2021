import itertools
import os
from typing import Tuple

import numpy as np
import copy
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskedConv(nn.Conv2d):
    def __init__(self, *args, type=1, is_conditional=False, n_groups=3, **kwargs):
        super().__init__(*args, **kwargs)
        # weight.shape = (out_channels, in_channels, conv_size, conv_size)
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self.set_mask(type, is_conditional, n_groups)

    def forward(self, x):
        return F.conv2d(x, self.mask * self.weight, bias=self.bias, stride=self.stride, padding=self.padding)

    def set_mask(self, type, is_conditional, n_groups):
        h, w = self.kernel_size
        out_channels, in_channels = self.weight.shape[:2]

        self.mask[:, :, h // 2, : w // 2] = 1
        self.mask[:, :, : h // 2] = 1
        if is_conditional:
            for g in range(n_groups):
                self.mask[out_channels // 3 * g:out_channels // 3 * (g + 1), :in_channels // 3 * (g + type), h // 2, w // 2] = 1
        if not is_conditional and type == 1:
            self.mask[:, :, h // 2, w // 2] = 1


class ResidualBlock(nn.Module):
    def __init__(self, inc, conv_size, is_conditional=False):
        super().__init__()
        h = inc // 2

        self.main = nn.Sequential(
            MaskedConv(in_channels=inc, out_channels=h, kernel_size=1, is_conditional=is_conditional),
            nn.ReLU(),
            MaskedConv(in_channels=h, out_channels=h, kernel_size=conv_size, padding=3, is_conditional=is_conditional),
            nn.ReLU(),
            MaskedConv(in_channels=h, out_channels=inc, kernel_size=1, is_conditional=is_conditional),
        )

        self.ln = LayerNorm(is_conditional, inc // 3 if is_conditional else inc)

    def forward(self, x):
        return self.ln(F.relu(self.main(x) + x))


class LayerNorm(nn.LayerNorm):
    def __init__(self, color_conditioning, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_conditioning = color_conditioning

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        if self.color_conditioning:
            x = x.contiguous().view(*(x_shape[:-1] + (3, -1)))
        x = super().forward(x)
        if self.color_conditioning:
            x = x.view(*x_shape)
        return x.permute(0, 3, 1, 2).contiguous()


class PixelCNN(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            n_classes: int,
            conv_size: int = 7,
            feature_maps: int = 64,
            n_convs: int = 4,
            is_conditional: bool = False,
    ):
        super().__init__()
        self.h, self.w, self.c = input_shape
        self.n_classes = n_classes
        self.is_conditional = is_conditional

        init_conv = [
            MaskedConv(self.c, feature_maps, conv_size, 1, 3, type=0, is_conditional=is_conditional),
            LayerNorm(is_conditional, feature_maps // 3),
            nn.ReLU(),
        ]

        hidden_convs = [ResidualBlock(feature_maps, conv_size, is_conditional=is_conditional) for _ in range(n_convs)]

        dense_convs = [
            MaskedConv(feature_maps, feature_maps, 1, is_conditional=is_conditional),
            nn.ReLU(),
        ]

        last_conv = [MaskedConv(feature_maps, self.c * n_classes, 1, is_conditional=is_conditional)]

        self.model = nn.Sequential(*(init_conv + hidden_convs + dense_convs + last_conv))

    def forward(self, x):
        out = (x / (self.n_classes - 1) - 0.5) / 0.5
        return self.model(out)

    def nll(self, x, y):
        # x.shape = (bs, c * n_classes, h, w)
        # y.shape = (bs, c, h, w)
        # wanna have (bs, n_classes, c, h, w)
        bs, c, h, w = y.shape
        if not self.is_conditional:
            x = x.reshape((bs, self.n_classes, c, h, w))
        else:
            x = x.view(bs, c, self.n_classes, h, w).permute(0, 2, 1, 3, 4)
        return F.cross_entropy(x, y)

    @torch.no_grad()
    def sample(self, n=100, device=DEVICE):
        self.eval()
        samples = torch.zeros(n, self.c, self.h, self.w).to(device)

        generation_bar = tqdm(total=self.c * self.h * self.w, desc="Generating")
        for i in range(self.h):
            for j in range(self.w):
                if not self.is_conditional:
                    logits = self.model(samples)
                    probs = (
                        logits.reshape((n, self.n_classes, self.c, self.h, self.w)).permute((0, 2, 3, 4, 1)).softmax(-1)
                    )

                for k in range(self.c):
                    if self.is_conditional:
                        logits = self.model(samples).view(n, self.c, self.n_classes, self.h, self.w).permute(0, 2, 1, 3, 4)
                        probs = logits[..., k, i, j].softmax(-1)

                    samples[:, k, i, j] = torch.multinomial(probs, 1).flatten()
                    generation_bar.update(1)

        generation_bar.close()
        return samples.cpu().detach().numpy().transpose(0, 2, 3, 1)


class Trainer:
    def __init__(self, n_epochs, lr, grad_clip = None, device=DEVICE):
        self.n_epochs = n_epochs
        self.device = device
        self.lr = lr
        self.grad_clip = grad_clip

    def fit(self, model, train_loader, val_loader):
        train_loss = []
        test_loss = []

        optimizer = opt.Adam(model.parameters(), lr=self.lr)

        model.to(self.device)
        test_loss.append(self.evaluate(model, val_loader))
        for epoch in range(self.n_epochs):
            model.train()
            tqdm_steps = tqdm(train_loader, desc=f"Training")
            for batch in tqdm_steps:
                loss = self.train_step(model, batch)

                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

                train_loss.append(loss.cpu().item())
                tqdm_steps.set_postfix(train_loss=train_loss[-1])

            test_loss.append(self.evaluate(model, val_loader))

        return train_loss, test_loss

    def train_step(self, model, batch):
        x, y = (batch, batch)
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.int64)

        logits = model(x)
        loss = model.nll(logits, y)

        return loss

    @torch.no_grad()
    def evaluate(self, model, val_loader):
        losses = []
        model.eval()
        for batch in val_loader:
            loss = self.train_step(model, batch)
            losses.append(loss.item())

        return np.mean(losses)


def q1_a(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, C) of samples with values in {0, 1, 2, 3}
    """
    print(len(train_data))
    batch_size = 128
    n_epochs = 1
    lr = 1e-3
    n_convs = 8
    feature_maps = 2 * 3 * 20
    device = torch.device("cpu")
    grad_clip = 1

    train_data = train_data.transpose(0, 3, 1, 2)
    test_data = test_data.transpose(0, 3, 1, 2)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=batch_size)

    model = PixelCNN(image_shape, n_classes=4, conv_size=7, n_convs=n_convs, feature_maps=feature_maps, is_conditional=True)

    trainer = Trainer(n_epochs=n_epochs, lr=lr, grad_clip=grad_clip, device=device)

    tr_loss, te_loss = trainer.fit(model, train_loader, val_loader)

    samples = model.sample(device=device)

    return tr_loss, te_loss, samples


if __name__ == "__main__":
    bs, h, w, c = 100, 32, 32, 3
    train_data = np.random.randint(0, 4, size=(bs, h, w, c))
    test_data = np.random.randint(0, 4, size=(bs, h, w, c))

    q1_a(train_data, test_data, (h, w, c), 1)

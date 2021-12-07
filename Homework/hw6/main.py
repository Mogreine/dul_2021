import itertools

import torch
import numpy as np
import torch.optim as opt
import torch.nn.functional as F

from typing import Tuple
from torch import nn
from torch.distributions import Normal, MultivariateNormal
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange


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


class MaskedResidualBlock(nn.Module):
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

        self.ln = LayerNorm(is_conditional, (inc // 3) if is_conditional else inc)

    def forward(self, x):
        return self.ln(F.relu(self.main(x) + x))


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.main(x)


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
            preprocessor: nn.Module,
            conv_size: int = 7,
            feature_maps: int = 64,
            n_convs: int = 4,
            is_conditional: bool = False,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.h, self.w, self.c = input_shape
        self.n_classes = n_classes
        self.is_conditional = is_conditional

        init_conv = [
            MaskedConv(self.c, feature_maps, conv_size, 1, 3, type=0, is_conditional=is_conditional),
            LayerNorm(is_conditional, feature_maps),
            nn.ReLU(),
        ]

        hidden_convs = [MaskedResidualBlock(feature_maps, conv_size, is_conditional=is_conditional) for _ in range(n_convs)]

        dense_convs = [
            MaskedConv(feature_maps, feature_maps, 1, is_conditional=is_conditional),
            nn.ReLU(),
        ]

        last_conv = [MaskedConv(feature_maps, self.c * n_classes, 1, is_conditional=is_conditional)]

        self.model = nn.Sequential(*(init_conv + hidden_convs + dense_convs + last_conv))

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def forward(self, x):
        with torch.no_grad():
            tokens = self.preprocessor.code_encoding(x)
            x = tokens.reshape(-1, 1, 8, 8)
        x = x.to(dtype=torch.float32)
        y = x.to(dtype=torch.int64)
        logits = self.model(x)
        loss = self.nll(logits, y)
        return {"loss": loss}

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
    def sample(self, n=100):
        self.eval()
        samples = torch.zeros(n, self.c, self.h, self.w).to(self.device)

        generation_bar = tqdm(total=self.c * self.h * self.w, desc="Generating")
        for i in range(self.h):
            for j in range(self.w):
                logits = self.model(samples).view(n, self.c, self.n_classes, self.h, self.w).permute(0, 2, 1, 3, 4)
                probs = logits[..., 0, i, j].softmax(-1)

                samples[:, 0, i, j] = torch.multinomial(probs, 1).flatten()
                generation_bar.update(1)

        generation_bar.close()
        return samples.squeeze(1)


class Trainer:
    def __init__(self, n_epochs, lr, grad_clip=None, device="cpu"):
        self.n_epochs = n_epochs
        self.device = device
        self.lr = lr
        self.grad_clip = grad_clip

    def fit(self, model, train_loader, val_loader):
        train_metrics = []
        test_metrics = []

        optimizer = opt.Adam(model.parameters(), lr=self.lr)

        model.to(self.device)
        test_metrics.append(self.evaluate(model, val_loader))
        for epoch in range(self.n_epochs):
            model.train()
            tqdm_steps = tqdm(train_loader, desc=f"Training")
            for batch in tqdm_steps:
                res = self._step(model, batch)
                loss = res["loss"]

                optimizer.zero_grad()
                loss.backward()
                # model.copy_grad()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

                train_metrics.append({k: v.detach().cpu().numpy() for k, v in res.items()})
                tqdm_steps.set_postfix(train_loss=loss.item())

            test_metrics.append(self.evaluate(model, val_loader))

        return train_metrics, test_metrics

    def _step(self, model, batch, return_rec=False):
        x = batch.to(self.device, dtype=torch.float32)
        res = model(x)
        return res

    @torch.no_grad()
    def evaluate(self, model, val_loader):
        model_out = []
        model.eval()
        for batch in val_loader:
            res = self._step(model, batch)
            model_out.append(res)

        res = {
            "loss": np.mean([el["loss"].cpu().numpy() for el in model_out]),
        }

        return res


class QEncoder(nn.Module):
    def __init__(self, size, code_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / size, 1.0 / size)

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        n, _, h, w = z.shape

        z_ = z.permute(0, 2, 3, 1).reshape(-1, self.code_dim)
        distances = (
            (z_ ** 2).sum(dim=1, keepdim=True)
            - 2 * torch.matmul(z_, self.embedding.weight.t())
            + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
        )
        encoding_indices = torch.argmin(distances, dim=1).reshape(n, h, w)
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2)

        return quantized, (quantized - z).detach() + z, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, latent_dim=16, beta=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 4, 2, 1),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        self.qencoder = QEncoder(64, 64)

        self.decoder = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

        print(sum([p.numel() for p in self.parameters()]))

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def code_encoding(self, batch):
        z = self.encoder(batch)
        return self.qencoder(z)[-1]

    @torch.no_grad()
    def code_decoding(self, batch):
        z = self.qencoder.embedding(batch).permute(0, 3, 1, 2)
        return self.decoder(z).permute(0, 2, 3, 1)

    def forward(self, x):
        z = self.encoder(x)
        e, e_, _ = self.qencoder(z)
        x_decoded = self.decoder(e_)

        batch_recon = x_decoded

        recon_loss = F.mse_loss(batch_recon, x)
        vq_loss = F.mse_loss(e, z.detach())  # is it working?
        commitment_loss = F.mse_loss(z, e.detach())

        return {"loss": recon_loss + vq_loss + self.beta * commitment_loss}

    @torch.inference_mode()
    def sample(self, prior_estimator, n=100):
        # bs, 8, 8
        tokens = prior_estimator.sample(n)
        tokens = tokens.to(dtype=torch.int64)

        x_dec = self.code_decoding(tokens)

        return x_dec

    @torch.inference_mode()
    def reconstruct(self, x):
        tokens = self.code_encoding(x).to(dtype=torch.int64)
        x_dec = self.code_decoding(tokens)

        return x_dec


def transform(x):
    return 2 * (np.transpose(x, (0, 3, 1, 2)) / 255.0).astype("float32") - 1


def transform_inverse(x):
    return (np.clip(x, -1, 1) * 0.5 + 0.5).astype("float32") * 255


def q2(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of VQ-VAE train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of VQ-VAE train losses evaluated once at initialization and after each epoch
    - a (# of training iterations,) numpy array of PixelCNN prior train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of PixelCNN prior train losses evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples with values in {0, ... 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in [0, 255]
    """
    batch_size = 128
    n_epochs = 1
    lr = 1e-3
    latent_dim = 16

    device = torch.device("cuda")
    grad_clip = None

    train_data = transform(train_data)
    test_data = transform(test_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=batch_size)

    vq_vae = VQVAE(latent_dim=latent_dim)

    trainer = Trainer(n_epochs=n_epochs, lr=lr, grad_clip=grad_clip, device=device)

    train_metrics, test_metrics = trainer.fit(vq_vae, train_loader, val_loader)
    train_metrics = np.vstack([list(el.values()) for el in train_metrics])
    test_metrics = np.vstack([list(el.values()) for el in test_metrics])

    prior_estimator = PixelCNN(
        input_shape=(8, 8, 1),
        n_classes=64,
        preprocessor=vq_vae,
    )

    train_metrics_prior, test_metrics_prior = trainer.fit(prior_estimator, train_loader, val_loader)
    train_metrics_prior = np.vstack([list(el.values()) for el in train_metrics_prior])
    test_metrics_prior = np.vstack([list(el.values()) for el in test_metrics_prior])


    samples = vq_vae.sample(prior_estimator).cpu().numpy()
    test_samples = next(iter(val_loader))[:50].to(device, dtype=torch.float32)
    test_samples_rec = vq_vae.reconstruct(test_samples)

    rec = torch.vstack([test_samples, test_samples_rec.permute(0, 3, 1, 2)]).permute(0, 2, 3, 1).cpu().numpy()

    return (
        train_metrics.flatten(),
        test_metrics.flatten(),
        train_metrics_prior.flatten(),
        test_metrics_prior.flatten(),
        transform_inverse(samples),
        transform_inverse(rec),
    )


if __name__ == "__main__":
    train_data = np.random.randint(0, 255, (500, 32, 32, 3), np.uint8)
    test_data = np.random.randint(0, 255, (500, 32, 32, 3), np.uint8)

    q2(train_data, test_data, 1)

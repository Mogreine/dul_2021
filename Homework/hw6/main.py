import itertools

import torch
import numpy as np
import torch.optim as opt
import torch.nn.functional as F

from torch import nn
from torch.distributions import Normal, MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


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
                model.copy_grad()
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


class VQVAE(nn.Module):
    def __init__(self, latent_dim=16, beta=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = nn.Sequential(
            *[
                nn.Conv2d(3, 32, 3, 2, 1),  # 16
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1),  # 8
            ]
        )

        self.decoder = nn.Sequential(
            *[
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),
            ]
        )

        self.embeddings = nn.Embedding(64, 64)

        print(sum([p.numel() for p in self.parameters()]))

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @torch.no_grad()
    def find_closest(self, x_enc):
        # bs, 64, 64 -> bs, 64 * 64, 64
        x_enc = x_enc.repeat_interleave(64, dim=1)

        # 64, 64 -> bs, 64 * 64, 64
        centroids = self.embeddings.weight.repeat(len(x_enc), 64, 1)

        # bs, 64 * 64, 64 -> bs, 64, 64, 64
        x_enc = x_enc.reshape(len(x_enc), 64, 64, 64)
        centroids = centroids.reshape(len(centroids), 64, 64, 64)

        # bs, 64, 64, 64 -> bs, 64
        res = torch.linalg.norm(x_enc - centroids, dim=-1).argmin(dim=-1)

        return res

    def encode(self, x):
        # encoding
        # bs, 64, 8, 8
        x_enc = self.encoder(x)

        # bs, 8, 8, 64
        x_enc = x_enc.transpose(1, 3)

        # bs, 64, 64
        x_enc = x_enc.reshape(len(x), 64, 64)

        # bs, 64
        tokens = self.find_closest(x_enc)

        return x_enc, tokens

    def decode(self, tokens):
        vq_emb = self.embeddings(tokens).reshape(len(tokens), 8, 8, 64).transpose(1, 3)
        x_dec = self.decoder(vq_emb)
        return x_dec

    def forward(self, x):
        bs = len(x)
        # tokens: bs, 64
        x_enc, tokens = self.encode(x)

        x_dec = self.decode(tokens)

        reconstruction_loss = 1 / 2 * (x - x_dec) ** 2
        reconstruction_loss = reconstruction_loss.reshape(bs, -1).sum(1).mean()

        # bs, 64, 64
        e = self.embeddings.weight[tokens]
        self.e = e
        self.x_enc = e + (x_enc - e).detach()

        loss = (
            reconstruction_loss
            + torch.linalg.norm(e - x_enc.detach(), dim=-1).sum(1).mean()
            + self.beta * torch.linalg.norm(e.detach() - x_enc, dim=-1).sum(1).mean()
        )

        return {"loss": loss}

    def copy_grad(self):
        self.x_enc = self.e + (self.x_enc - self.e).detach()

    @torch.inference_mode()
    def sample(self, n=100):
        z = self.p_z.sample((n,))
        mu_z = self.decoder_linear(z).reshape(n, 256, 1, 1)
        mu_z = self.decoder(mu_z)

        return mu_z

    @torch.inference_mode()
    def reconstruct(self, x):
        x_enc, tokens = self.encode(x)
        x_dec = self.decode(tokens)

        return x_dec


def transform(x):
    return 2 * (np.transpose(x, (0, 3, 1, 2)) / 255.0).astype("float32") - 1


def transform_inverse(x):
    return (np.transpose(np.clip(x, -1, 1), (0, 2, 3, 1)) * 0.5 + 0.5).astype("float32") * 255


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

    model = VQVAE(latent_dim=latent_dim)

    trainer = Trainer(n_epochs=n_epochs, lr=lr, grad_clip=grad_clip, device=device)

    train_metrics, test_metrics = trainer.fit(model, train_loader, val_loader)
    train_metrics = np.vstack([list(el.values()) for el in train_metrics])
    test_metrics = np.vstack([list(el.values()) for el in test_metrics])

    samples = model.sample().cpu().numpy()
    test_samples = next(iter(val_loader))[:50].to(device, dtype=torch.float32)
    test_samples_rec = model.reconstruct(test_samples)

    rec = torch.vstack([test_samples, test_samples_rec]).cpu().numpy()

    return (
        train_metrics,
        test_metrics,
        transform_inverse(samples),
        transform_inverse(rec),
    )


if __name__ == "__main__":
    train_data = np.random.randint(0, 255, (500, 32, 32, 3), np.uint8)
    test_data = np.random.randint(0, 255, (500, 32, 32, 3), np.uint8)

    q2(train_data, test_data, 1)

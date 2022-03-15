import torch
import numpy as np

from torch import nn
from torch.distributions import Normal
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import trange, tqdm


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = (
            torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width, s_depth)
        )
        output = output.permute(0, 3, 1, 2)
        return output


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


# Spatial Upsampling with Nearest Neighbors
class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.dts = DepthToSpace(2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        _x = torch.cat([x, x, x, x], dim=1)
        _x = self.dts(_x)
        _x = self.conv(_x)
        return _x


# Spatial Downsampling with Spatial Mean Pooling
class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True):
        super().__init__()
        self.std = SpaceToDepth(2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        _x = self.std(x)
        _x = sum(_x.chunk(4, dim=1)) / 4.0
        _x = self.conv(_x)
        return _x


class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim, n_filters=256, kernel_size=(3, 3)):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.ReLU(),
        )
        self.residual = Downsample_Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=1)
        self.shortcut = Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        x_ = self.seq(x)
        return self.residual(x_) + self.shortcut(x)


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.n_filters = n_filters

        self.seq = nn.Sequential(
            nn.BatchNorm2d(self.in_dim),
            nn.ReLU(),
            nn.Conv2d(self.in_dim, self.n_filters, self.kernel_size, padding=1),
            nn.BatchNorm2d(self.n_filters),
            nn.ReLU(),
        )

        self.residual = Upsample_Conv2d(self.n_filters, self.n_filters, self.kernel_size, padding=1)
        self.shortcut = Upsample_Conv2d(self.in_dim, self.n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        _x = self.seq(x)
        residual = self.residual(_x)
        shortcut = self.shortcut(x)
        return residual + shortcut


class Generator(nn.Module):
    def __init__(self, n_filters=128, z_dim=128):
        super().__init__()
        self.n_filters = n_filters
        self.z_dim = z_dim

        self.fc = nn.Linear(128, 4 * 4 * n_filters)

        self.seq = nn.Sequential(
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh(),
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dist(self):
        return Normal(torch.tensor([0.], device=self.device), torch.tensor([1.], device=self.device))

    def forward(self, z):
        out = self.fc(z).reshape(-1, self.n_filters, 4, 4)
        return self.seq(out)

    def sample(self, n_samples=100):
        z = self.dist.sample([n_samples, self.z_dim]).to(self.device).squeeze(-1)
        return self(z)


class Critic(nn.Module):
    def __init__(self, n_filters=256):
        super().__init__()
        self.main = nn.Sequential(
            ResnetBlockDown(3, n_filters),
            ResnetBlockDown(n_filters, n_filters),
            ResnetBlockDown(n_filters, n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(4, 4), padding=0),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc(self.main(x).squeeze())


class SNGAN(nn.Module):
    def __init__(self, n_filters=128, lr=2e-4, n_critic=5, lam=10.0):
        super().__init__()
        self.n_critic = n_critic
        self.lam = lam

        self.generator = Generator(n_filters)
        self.critic = Critic(n_filters)

        self.g_optim = Adam(self.generator.parameters(), lr=lr, betas=(0, 0.9))
        self.c_optim = Adam(self.critic.parameters(), lr=lr, betas=(0, 0.9))

    @property
    def device(self):
        return next(self.parameters()).device

    def __critic_loss(self, real, fake):
        score_real = self.critic(real)
        score_fake = self.critic(fake)
        return -score_real.mean() + score_fake.mean() + self.lam * self.__gradient_penalty(real, fake)

    def __gradient_penalty(self, real, fake):
        """
        see algo 1 from https://arxiv.org/pdf/1704.00028.pdf
        """
        bs = real.shape[0]
        eps = torch.rand(bs, 1, 1, 1).to(self.device).expand_as(real)
        interps = eps * real + (1 - eps) * fake

        scores = self.critic(interps)

        grads = torch.autograd.grad(scores, interps, torch.ones_like(scores, device=self.device), create_graph=True)[
            0
        ].reshape(bs, -1)

        return torch.mean((torch.norm(grads, dim=1) - 1) ** 2)

    def fit(self, trainloader, n_iter):
        losses = []
        total_iters = 0
        # epochs = n_iter // len(trainloader)
        epochs = n_iter
        # epochs = self.n_critic * n_iter // len(trainloader)

        g_scheduler = LambdaLR(self.g_optim, lambda epoch: (epochs - epoch) / epochs, last_epoch=-1)
        c_scheduler = LambdaLR(self.c_optim, lambda epoch: (epochs - epoch) / epochs, last_epoch=-1)

        for epoch in trange(epochs, desc="Training...", leave=False):
            for batch_real in tqdm(trainloader, desc="Batch", leave=False):
                total_iters += 1

                batch_real = batch_real.to(self.device)
                batch_fake = self.generator.sample(batch_real.shape[0])

                critic_loss = self.__critic_loss(batch_real, batch_fake)

                self.c_optim.zero_grad()
                critic_loss.backward()
                self.c_optim.step()

                losses.append(critic_loss.detach().cpu().numpy())

                if total_iters % self.n_critic == 0:
                    g_loss = -self.critic(self.generator.sample(batch_real.shape[0])).mean()

                    self.g_optim.zero_grad()
                    g_loss.backward()
                    self.g_optim.step()

            g_scheduler.step()
            c_scheduler.step()

        return np.array(losses)

    @torch.no_grad()
    def sample(self, n):
        return self.generator.sample(n)


def convert_forward(x):
    return torch.tensor((x - 0.5) * 2, dtype=torch.float32)


def convert_backward(x):
    return (x / 2 + 0.5).permute(0, 2, 3, 1).detach().cpu().numpy()


def q1(train_data):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of WGAN critic train losses evaluated every minibatch
    - a (1000, 32, 32, 3) numpy array of samples from your model in [0, 1].
        The first 100 will be displayed, and the rest will be used to calculate the Inception score.
    """
    n_epochs = 2
    bs = 256
    device = "cpu"

    train_data = convert_forward(train_data)
    train_dl = DataLoader(train_data, batch_size=bs, shuffle=True)

    gan = SNGAN()
    gan.to(device)
    losses = gan.fit(train_dl, n_epochs)
    samples = gan.sample(1000)
    samples = convert_backward(samples)

    return losses, samples


if __name__ == "__main__":
    train_data = torch.randn(10, 3, 32, 32)
    losses, samples = q1(train_data)

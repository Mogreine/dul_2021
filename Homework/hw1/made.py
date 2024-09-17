import torch
import numpy as np
import torch.nn.functional as F
import itertools

from torch import nn
from typing import List

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange


class LinearMasked(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))


class MADE(nn.Module):
    def __init__(self, seq_len, n_classes, n_layers, hidden_dim, seed=42):
        super().__init__()
        self.seed = seed
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.embedding = nn.Embedding(n_classes, n_classes)

        first_layer = [LinearMasked(n_classes * seq_len, hidden_dim), nn.ReLU()]
        hidden_layers = list(
            itertools.chain(*[[LinearMasked(hidden_dim, hidden_dim), nn.ReLU()] for _ in range(n_layers - 1)])
        )
        last_layer = [LinearMasked(hidden_dim, seq_len * n_classes)]

        self.mlp = nn.Sequential(*(first_layer + hidden_layers + last_layer))

        self.update_masks()

    def update_masks(self):
        L = self.n_layers

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % L

        self.m = {}
        self.m[-1] = np.repeat(np.arange(self.seq_len), self.n_classes)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.seq_len, size=self.hidden_dim)

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # masks[-1] = np.concatenate([masks[-1]] * self.n_classes, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.modules() if isinstance(l, LinearMasked)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        x = self.embedding(x)
        # x = F.one_hot(x, self.n_classes).to(torch.float32)
        x = x.reshape((len(x), -1))
        return self.mlp(x)

    def nll(self, x, y):
        # x.shape = (bs, n_classes * in_dim)
        # y.shape = (bs, in_dim)
        bs, in_dim = y.shape
        # x = x.reshape((bs, self.n_classes, in_dim))
        return F.cross_entropy(x.reshape(bs * self.seq_len, -1), y.flatten())


def create_dataloader(
        raw_data: List[List[int]],
        batch_size: int,
        shuffle: bool = True,
) -> DataLoader:
    # raw_data, shape = (n, n_classes)
    ds = TensorDataset(torch.tensor(raw_data, dtype=torch.int64).reshape(len(raw_data), -1))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return dl


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    losses = []
    for batch in dataloader:
        loss = train_step(model, batch, device)
        losses.append(loss.item())

    return np.mean(losses)


def train_step(model, batch, device):
    batch = batch[0]
    x, y = (batch, batch)
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    loss = model.nll(logits, y)

    return loss


@torch.no_grad()
def get_joint_dist(model, d, device="cpu"):
    model.eval()
    idx = itertools.product(range(d), range(d))
    res = np.ones((d, d))

    for i, j in idx:
        batch = torch.tensor([[i, j]], device=device)

        logits = model(batch)
        probs1 = logits[:, :d].softmax(dim=-1)[:, i]
        probs2 = logits[:, d:].softmax(dim=-1)[:, j]

        res[i, j] = probs1 * probs2

    return res


@torch.no_grad()
def generate_img(model, H, W, n_samples, n_classes, device="cpu"):
    model.eval()
    model.to(device)
    res = torch.zeros((n_samples, H * W), device=device, dtype=torch.int64)
    for i in range(H):
        for j in range(W):
            imgs = model(res)
            # res.shape = (bs, H * W * n_classes)
            imgs = imgs.reshape((-1, n_classes)).softmax(-1).multinomial(1).flatten()
            imgs = imgs.reshape(n_samples, H, W)
            res[:, i * W + j] = imgs[:, i, j]

    return res.reshape((n_samples, H, W, 1)).cpu().numpy()


def q1_b(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: An (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """
    device = torch.device("cuda")
    batch_size = 64
    n_epochs = 20
    lr = 1e-3

    train_dl = create_dataloader(train_data, batch_size)
    test_dl = create_dataloader(test_data, batch_size, False)

    h, w = image_shape
    model = MADE(seq_len=h * w, n_classes=2, hidden_dim=128, n_layers=4)

    train_losses = []
    test_losses = [evaluate(model, test_dl, device)]

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    for ep in trange(n_epochs, desc="Epochs"):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()

            loss = train_step(model, batch, device)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        test_losses.append(evaluate(model, test_dl, device))

    imgs = generate_img(model, h, w, 100, 2, device)

    return np.array(train_losses), np.array(test_losses), imgs


def q1_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train, 2) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test, 2) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for each random variable x1 and x2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d, d) of probabilities (the learned joint distribution)
    """
    device = torch.device("cpu")
    batch_size = 1024
    n_epochs = 20
    lr = 1e-3

    train_dl = create_dataloader(train_data, batch_size)
    test_dl = create_dataloader(test_data, batch_size, False)

    model = MADE(seq_len=2, n_classes=d, hidden_dim=128, n_layers=4)

    train_losses = []
    test_losses = [evaluate(model, test_dl, device)]

    model.to(device)
    print(len(list(model.parameters())))
    optimizer = Adam(model.parameters(), lr=lr)
    for ep in trange(n_epochs, desc="Epochs"):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()

            loss = train_step(model, batch, device)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        test_losses.append(evaluate(model, test_dl, device))

    dist = get_joint_dist(model, d, device)

    return np.array(train_losses), np.array(test_losses), dist


if __name__ == "__main__":
    model = MADE(2, 25, 3, 64)

    batch = torch.tensor([[1, 2], [3, 4]])

    logits = model(batch)
    loss = model.nll(logits, batch)

    print(loss)
    train_data = [
        [3, 4],
        [3, 4],
        [3, 4],
        [3, 4],
        [3, 4],
        [3, 4],
        [3, 4],
        [3, 4],
        [3, 4],
        [3, 4],
        [3, 4],
    ]

    test_data = [
        [2, 4],
        [2, 4],
        [2, 4],
        [2, 4],
        [2, 4],
        [2, 4],
        [2, 4],
        [2, 4],
        [2, 4],
        [2, 4],
    ]

    q1_a(train_data, test_data, 25, 1)

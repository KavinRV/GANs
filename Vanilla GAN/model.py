# Vanilla GAN model

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
# from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, lat_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.gentr = nn.Sequential(
            nn.Linear(lat_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim*2, out_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gentr(x)


class Discriminator(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.discmtr = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.discmtr(x)


# Set Hyperparameters
lr = 0.0003
batch_size = 32
epochs = 50
lat_dim = 64
img_dim = 784
display_step = 500
device = 'cuda' if torch.cuda.is_available() else "mps"

discmtr = Discriminator().to(device)
genrtr = Generator(lat_dim, img_dim).to(device)
fixed_noise = torch.randn((batch_size, lat_dim)).to(device)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_discmtr = optim.Adam(discmtr.parameters(), lr=lr)
opt_genrtr = optim.Adam(genrtr.parameters(), lr=lr)
criterion = nn.BCELoss()

writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, lat_dim).to(device)
        fake = genrtr(noise)
        discmtr_real = discmtr(real).view(-1)
        lossD_real = criterion(discmtr_real, torch.ones_like(discmtr_real))
        discmtr_fake = discmtr(fake).view(-1)
        lossD_fake = criterion(discmtr_fake, torch.zeros_like(discmtr_fake))
        lossD = (lossD_real + lossD_fake) / 2
        discmtr.zero_grad()
        lossD.backward(retain_graph=True)
        opt_discmtr.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = discmtr(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        genrtr.zero_grad()
        lossG.backward()
        opt_genrtr.step()

        if batch_idx == 0:
            print(f'Epoch [{epoch}/{epochs}] Loss D: {lossD:.4f}, loss G: {lossG:.4f}')

            with torch.no_grad():
                fake = genrtr(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image('MNIST Fake Images', img_grid_fake, global_step=step)
                writer_real.add_image('MNIST Real Images', img_grid_real, global_step=step)
                step += 1

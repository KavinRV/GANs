import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
lr = 2e-4
batch_size = 256
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 25
features_d = 64
features_g = 64
my_transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=my_transforms, download=True)
# dataset = datasets.ImageFolder(root="dataset/", transform=my_transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gentr = Generator(z_dim, channels_img, features_g).to(device)
discmtr = Discriminator(channels_img, features_d).to(device)
initialize_weights(gentr)
initialize_weights(discmtr)

opt_g = optim.Adam(gentr.parameters(), lr=lr, betas=(0.5, 0.999))
opt_d = optim.Adam(discmtr.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gentr.train()
discmtr.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gentr(noise)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = discmtr(real).reshape(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discmtr(fake).reshape(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        discmtr.zero_grad()
        lossD.backward(retain_graph=True)
        opt_d.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = discmtr(fake).reshape(-1)
        lossG = criterion(output, torch.ones_like(output))
        gentr.zero_grad()
        lossG.backward()
        opt_g.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gentr(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            step += 1


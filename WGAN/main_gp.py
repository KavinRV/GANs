# WGAN with gradient penalty training

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Critic, Generator, initialize_weights

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
lr = 2e-4
batch_size = 128
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 25
n_critics = 5
features_c = 64
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

dataset = datasets.MNIST(root="../dataset/", transform=my_transforms, download=True)
# dataset = datasets.ImageFolder(root="dataset/", transform=my_transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gentr = Generator(z_dim, channels_img, features_g).to(device)
critic = Critic(channels_img, features_c).to(device)
initialize_weights(gentr)
initialize_weights(critic)

opt_g = optim.Adam(gentr.parameters(), lr=lr, betas=(0, 0.9))
opt_c = optim.Adam(critic.parameters(), lr=lr, betas=(0, 0.9))


fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gentr.train()
critic.train()


def gradient_penalty(critic, real, fake, device="cpu"):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real.shape[0], 1, 1, 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    grad_pen = torch.mean((gradient_norm - 1) ** 2)
    return grad_pen


for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        #               min -E[critic(real)] + E[critic(fake)]
        # for _ in range(n_critics):
        noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
        fake = gentr(noise)

        critic_real = critic(real).reshape(-1)
        critic_fake = critic(fake).reshape(-1)
        gp = gradient_penalty(critic, real, fake, device=device)
        loss_c = (
            -(torch.mean(critic_real) - torch.mean(critic_fake)) + gp * 10
        )

        critic.zero_grad()
        loss_c.backward(retain_graph=True)
        opt_c.step()
        if batch_idx%5 != 0:
            continue

        # Train Generator: min -E[critic(gen_fake)] <-> max E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_g = -torch.mean(gen_fake)

        gentr.zero_grad()
        loss_g.backward()
        opt_g.step()

        if batch_idx%100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_c:.4f}, loss G: {loss_g:.4f}"
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

                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
            step += 1






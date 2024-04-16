import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 4 * 4 * 512)
        self.conv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, 512, 4, 4)
        out = F.relu(out)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = torch.tanh(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.fc = nn.Linear(256 * 4 * 4, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv3(out)
        out = F.leaky_relu(out, 0.2)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=True)

latent_dim = 100
num_epochs = 100
lr = 0.0002
beta1 = 0.5

generator = Generator(latent_dim)
discriminator = Discriminator()

generator.apply(weights_init)
discriminator.apply(weights_init)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, latent_dim).to(device)

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)

        # Train the discriminator
        optimizer_D.zero_grad()

        # Sample noise for the generator
        noise = torch.randn(real_images.size(0), latent_dim).to(device)

        # Generate fake images
        fake_images = generator(noise)

        # Classify real images
        real_labels = torch.ones(real_images.size(0)).to(device).unsqueeze(1)
        real_output = discriminator(real_images)
        real_loss = F.binary_cross_entropy_with_logits(real_output, real_labels)

        # Classify fake images
        fake_labels = torch.zeros(fake_images.size(0)).to(device).unsqueeze(1)
        fake_output = discriminator(fake_images)
        fake_loss = F.binary_cross_entropy_with_logits(fake_output, fake_labels)

        # Backpropagate the loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train the generator
        optimizer_G.zero_grad()

        # Generate fake images
        noise = torch.randn(real_images.size(0), latent_dim).to(device)
        fake_images = generator(noise)
# Classify fake images as real
        real_labels = torch.ones(fake_images.size(0)).to(device).unsqueeze(1)
        fake_output = discriminator(fake_images)
        g_loss = F.binary_cross_entropy_with_logits(fake_output, real_labels)

        # Backpropagate the loss
        g_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    # Save generated images every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake_images = generator(fixed_noise)
            torchvision.utils.save_image(fake_images, "C:/Users/emircan.karakus/PycharmProjects/GANs_1/generated_images/epoch_{}.png".format(epoch + 1), nrow=8, normalize=True)


            ##The output shows the loss values for the discriminator (D_loss) and the generator (G_loss) at two different steps during the first epoch of training. Specifically, it shows the losses after the first step and after the 100th step.

            ##At each step, the generator creates a batch of fake images, and the discriminator is trained to distinguish between the real and fake images. The generator is then trained to try to fool the discriminator. The losses reflect how well the discriminator and generator are performing at each step.



            ##The D_loss represents how well the discriminator is able to distinguish between real and fake images. A lower D_loss indicates that the discriminator is doing a better job of distinguishing between real and fake images.

            ##The G_loss represents how well the generator is able to fool the discriminator. A lower G_loss indicates that the generator is doing a better job of creating images that look like they come from the real dataset.


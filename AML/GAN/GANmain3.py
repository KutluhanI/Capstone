import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision import datasets, transforms
from IPython import display
import progressbar

# Constants
OUTPUT_DIR = './data/CIFAR-10'
BATCH_SIZE = 100
LR = 0.0002
NUM_EPOCHS = 0
NUM_TEST_SAMPLES = 32
FGSM_EPSILON = 0.007

# Load CIFAR-10 dataset
def load_data():
    compose = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return datasets.CIFAR10(root=OUTPUT_DIR, train=False, transform=compose, download=True)

data = load_data()
data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
NUM_BATCHES = len(data_loader)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 1024 * 4 * 4)
        return self.output(x)

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(100, 1024 * 4 * 4)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 1024, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x

# Initialize weights
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)

# Create noise for generator input
def noise(size):
    n = torch.randn(size, 100)
    return n.cuda() if torch.cuda.is_available() else n

# Real and fake data targets
def real_data_target(size):
    return torch.ones(size, 1).cuda() if torch.cuda.is_available() else torch.ones(size, 1)

def fake_data_target(size):
    return torch.zeros(size, 1).cuda() if torch.cuda.is_available() else torch.zeros(size, 1)

# FGSM Attack
def fgsm_attack(data, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    return torch.clamp(perturbed_data, 0, 1)

# Train Discriminator
def train_discriminator(optimizer, real_data, fake_data, model):
    optimizer.zero_grad()
    prediction_real = model(real_data)
    error_real = criterion(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    data_grad = real_data.grad.data
    perturbed_real_data = fgsm_attack(real_data, FGSM_EPSILON, data_grad)

    prediction_real_adv = model(perturbed_real_data)
    error_real_adv = criterion(prediction_real_adv, real_data_target(real_data.size(0)))
    error_real_adv.backward()

    prediction_fake = model(fake_data)
    error_fake = criterion(prediction_fake, fake_data_target(fake_data.size(0)))
    error_fake.backward()
    optimizer.step()
    return error_real + error_real_adv + error_fake, prediction_real, prediction_fake

# Train Generator
def train_generator(optimizer, fake_data, model):
    optimizer.zero_grad()
    prediction = model(fake_data)
    error = criterion(prediction, real_data_target(prediction.size(0)))
    error.backward()
    optimizer.step()
    return error

# Generate and display images
def generate_images(test_images, num_images, normalize=True):
    images = test_images
    horizontal_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
    nrows = int(np.sqrt(num_images))
    grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)
    plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
    plt.axis('off')
    display.display(plt.gcf())
    plt.close()
    plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
    plt.axis('off')
    plt.close()

# Initialize models
generator = Generator()
discriminator = Discriminator()
generator.apply(init_weights)
discriminator.apply(init_weights)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Test noise for generating images
test_noise = noise(NUM_TEST_SAMPLES)

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    progress_bar = progressbar.ProgressBar()
    d_running_loss = 0
    g_running_loss = 0

    for n_batch, (real_batch, _) in enumerate(progress_bar(data_loader)):
        real_data = Variable(real_batch, requires_grad=True).cuda() if torch.cuda.is_available() else Variable(real_batch, requires_grad=True)
        fake_data = generator(noise(real_data.size(0))).detach()
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data, discriminator)

        fake_data = generator(noise(real_batch.size(0)))
        g_error = train_generator(g_optimizer, fake_data, discriminator)

        d_running_loss += d_error.item()
        g_running_loss += g_error.item()

    print(f"Loss (Discriminator): {d_running_loss}")
    print(f"Loss (Generator): {g_running_loss}")

    test_images = generator(test_noise).data
    generate_images(test_images, NUM_TEST_SAMPLES)

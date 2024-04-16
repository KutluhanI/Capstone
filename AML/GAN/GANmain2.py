from IPython import display
import numpy as np
from matplotlib import pyplot as plt
import progressbar

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision import transforms, datasets

OUTPUT_DIR = './data/CIFAR-10'
BATCH_SIZE = 100
LR = 0.001
NUM_EPOCHS = 200
NUM_TEST_SAMPLES = 32


def load_data():
    compose = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    return datasets.CIFAR10(root=OUTPUT_DIR, train=False, transform=compose, download=True)


data = load_data()
# data = torch.utils.data.Subset(data, [0, 5000])
data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
NUM_BATCHES = len(data_loader)

print("No. of Batches = ", NUM_BATCHES)


class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024), nn.LeakyReLU(0.2, inplace=True))
        self.output = nn.Sequential(nn.Linear(1024 * 4 * 4, 1), nn.Sigmoid())

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Flatten and apply sigmoid
        x = x.view(-1, 1024 * 4 * 4)
        x = self.output(x)
        return x


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.linear = torch.nn.Linear(100, 1024 * 4 * 4)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False))
        self.output = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 4, 4)

        # Convolutional layers
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)

        # Apply Tanh
        return self.output(x)

def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available():
      return n.cuda()
    return n
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

generator = Generator()
discriminator = Discriminator()
generator.apply(init_weights)
discriminator.apply(init_weights)

# Enable cuda if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

# d_optimizer = optim.SGD(discriminator.parameters(), lr=LR, momentum=0.9)
# g_optimizer = optim.SGD(generator.parameters(), lr=LR, momentum=0.9)
d_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))

# Loss function
# criterion = nn.MSELoss()
criterion = nn.BCELoss()

## Use Adam Optimizer with BCELoss function

def real_data_target(size):
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data


def fake_data_target(size):
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data


def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = criterion(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = criterion(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = criterion(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

test_noise = noise(NUM_TEST_SAMPLES)


def generate_images(test_images, num_images, normalize=True):
    images = test_images

    # Make horizontal grid from image tensor
    horizontal_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
    # Make vertical grid from image tensor
    nrows = int(np.sqrt(num_images))
    grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)

    # Plot and save horizontal
    fig = plt.figure(figsize=(16, 16))
    plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
    plt.axis('off')
    if True:
        display.display(plt.gcf())
    plt.close()

    # Save squared
    fig = plt.figure()
    plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
    plt.axis('off')
    plt.close()


for epoch in range(NUM_EPOCHS):
    print("\nEpoch #", epoch, "in progress...")
    progress_bar = progressbar.ProgressBar()
    d_running_loss = 0
    g_running_loss = 0

    for n_batch, (real_batch, _) in enumerate(progress_bar(data_loader)):
        print("\n yoyo #", n_batch, "in progress...")
        # 1. Train Discriminator
        #     inputs, _ = real_batch
        real_data = Variable(real_batch)
        if torch.cuda.is_available():
            real_data = real_data.cuda()
        # Generate fake data
        fake_data = generator(noise(real_data.size(0))).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)

        d_running_loss += d_error.item()
        g_running_loss += g_error.item()

    #     loss = criterion(outputs, inputs)
    print("Loss (Discriminator):", d_running_loss)
    print("Loss (Generator):", g_running_loss)

    test_images = generator(test_noise).data.cpu()
    generate_images(test_images, NUM_TEST_SAMPLES)

num_samples = NUM_TEST_SAMPLES
test_noise = noise(num_samples)
test_images = generator(test_noise).data.cpu()
generate_images(test_images, NUM_TEST_SAMPLES)
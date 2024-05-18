import torchvision.models as models
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the pretrained ResNet-18 model
net = models.resnet18(weights=False, num_classes=10)
net.load_state_dict(torch.load('image_recognition_model.pth', map_location=torch.device('cpu')))
net.eval()

def fgsm_attack(data, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

def generate_fgsm_adversarial_example(model, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = fgsm_attack(data, epsilon, data_grad)
    return perturbed_data

# Data transformation and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Parameters
epsilon = 0.007
num_images = 100

# Counters
correct_original = 0
correct_adversarial = 0

for i, (data, target) in enumerate(test_loader):
    if i >= num_images:
        break

    # Move data and target to CPU (if not already)
    data, target = data.to('cpu'), target.to('cpu')

    # Original prediction
    output = net(data)
    original_pred = output.argmax(dim=1, keepdim=True)
    correct_original += original_pred.eq(target.view_as(original_pred)).sum().item()

    # Generate adversarial example
    adversarial_data = generate_fgsm_adversarial_example(net, data, target, epsilon)

    # Adversarial prediction
    output = net(adversarial_data)
    adversarial_pred = output.argmax(dim=1, keepdim=True)
    correct_adversarial += adversarial_pred.eq(target.view_as(adversarial_pred)).sum().item()

    print(f'Image {i+1}/{num_images}')
    print('Original prediction:', original_pred.item())
    print('Adversarial prediction:', adversarial_pred.item())

# Calculate accuracy
accuracy_original = correct_original / num_images
accuracy_adversarial = correct_adversarial / num_images

print(f'\nAccuracy on original images: {accuracy_original * 100:.2f}%')
print(f'Accuracy on adversarial images: {accuracy_adversarial * 100:.2f}%')

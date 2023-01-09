import torch
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

def showImages(fake, real):
    transform = torchvision.transforms.ToPILImage()
    f = fake.detach()
    r = real.detach()
    plt.figure(figsize=(20,20))
    columns = 5
    fakes = f[:5]
    for i, image in enumerate(fakes):
        plt.subplot(int(len(fakes) / columns + 1), columns, i + 1)
        plt.imshow(transform(image))

    reals = r[:5]
    for i, image in enumerate(reals):
        plt.subplot(1+int(len(reals) / columns + 1), columns, i + 1)
        plt.imshow(transform(image))



def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0, 0.02)
        torch.nn.init.normal_(m.bias, 0, 0.02)

def UniformNoise(batch_size: int, noise_dim: int):
    rand = torch.rand(batch_size, noise_dim, 1, 1)
    return (2*rand)-1

def NormalNoise(batch_size: int, noise_dim: int):
    return torch.randn(batch_size, noise_dim, 1, 1)
    

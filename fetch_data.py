import torchvision.datasets as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def getLFW(root: str = '.', batch_size: int = 100, shuffle: bool = False) -> None:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((130, 130)),
        transforms.Resize((64, 64))
    ])
    train = data.LFWPeople(root=root, transform=transform, download=True)
    train_loader = DataLoader(train, batch_size, shuffle, drop_last=True)
    return train_loader

def getMNIST(root: str = '.', batch_size: int = 100, shuffle: bool = False) -> None:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])
    train = data.MNIST(root=root, train=False, transform=transform, download=True)
    train_loader = DataLoader(train, batch_size, shuffle, drop_last=True)
    return train_loader

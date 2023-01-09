import torch

class Discriminator(torch.nn.Module):
    def __init__(self, img_dim) -> None:
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=img_dim, out_channels=128, kernel_size=4, stride=4, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(512)
        self.conv4 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(1024)
        self.conv5 = torch.nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=1)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        pass

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.leakyrelu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.leakyrelu(x)
        x = self.bn4(x)
        x = self.conv5(x)
        return x

class Generator(torch.nn.Module):
    def __init__(self, noise_dim, img_dim) -> None:
        super(Generator, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=noise_dim, out_channels=1024, kernel_size=4, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.conv3 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=img_dim, kernel_size=4, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        pass

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.tanh(x)
        return x
import torch
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image
try:
    import clip
except:
    import os
    os.system("pip install git+https://github.com/openai/CLIP.git")
    import clip


class PretrainedResnetBinaryClassifier(nn.Module):
    # Deprecated!
    def __init__(self, n_layers, pretrained=True):
        super().__init__()
        assert n_layers in [18, 34, 50, 101, 152]
        self.f = getattr(torchvision.models, f"resnet{n_layers}")(pretrained=pretrained)

        # replace the last fully connected layer to a flattening one
        self.out_dim = self.f.fc.in_features
        self.f.fc = nn.Flatten()
        self.linear = nn.Linear(self.out_dim, 1)

    def forward(self, inputs):
        x = self.f(inputs)
        x = self.linear(x)
        x = torch.sigmoid(x).squeeze()
        return x


class PretrainedResnetClassifier(nn.Module):
    def __init__(self, n_layers, n_heads, pretrained=True):
        super().__init__()
        assert n_layers in [18, 34, 50, 101, 152]

        self.f = getattr(torchvision.models, f"resnet{n_layers}")(pretrained=pretrained)
        self.n_heads = n_heads

        # replace the last fully connected layer to a flattening one
        self.out_dim = self.f.fc.in_features
        self.f.fc = nn.Flatten()
        self.class_heads = nn.Linear(self.out_dim, self.n_heads)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.f(inputs)
        x = self.class_heads(x)
        return self.sigmoid(x).T # of shape [n_heads, batch_size]

    def get_num_heads(self):
        return self.n_heads

    # batch is a list of StackedImgs
    def preprocess(self, batch, device='cpu'):
        tensorize = transforms.ToTensor()
        return torch.stack([tensorize(img.open()) for img in batch], dim=0).to(device)


class PretrainedCLIPResnet(nn.Module):
    def __init__(self, n_layers, n_heads):
        super().__init__()
        assert n_layers == 50
        self.n_heads = n_heads
        self.clip, self.preprocessor = clip.load('RN50', device='cpu')
        self.clip = self.clip.visual
        out_dim = self.clip.attnpool.c_proj.out_features
        self.class_heads = nn.Linear(out_dim, n_heads)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.clip(inputs)
        x = self.class_heads(x)
        return self.sigmoid(x).T # of shape [n_heads, batch_size]

    def get_num_heads(self):
        return self.n_heads

    def preprocess(self, batch, device='cpu'):
        return torch.stack([self.preprocessor(img.open()) for img in batch], dim=0).to(device)

class PretrainedCLIPViT(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.clip, self.preprocessor = clip.load('ViT-B/32', device='cpu')
        self.clip = self.clip.visual
        out_dim = 512
        self.class_heads = nn.Linear(out_dim, n_heads)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.clip(inputs)
        x = self.class_heads(x)
        return self.sigmoid(x).T

    def get_num_heads(self):
        return self.n_heads

    def preprocess(self, batch, device='cpu'):
        return torch.stack([self.preprocessor(img.open()) for img in batch], dim=0).to(device)

class PretrainedDensenet(nn.Module):
    def __init__(self, n_heads=2, pretrained=True):
        super().__init__()
        self.n_heads = n_heads
        self.network = torchvision.models.densenet121(pretrained=True)
        # replace the classifier head on densenet with our own heads
        penult_dim = self.network.classifier.in_features
        self.class_heads = nn.Sequential(
            nn.Linear(penult_dim, n_heads),
            nn.Sigmoid()
            )
        self.network.classifier = self.class_heads

        # setting up proprocessor
        mean = [0.485, 0.456, 0.406] # mean, std over imagenet images
        std = [0.229, 0.224, 0.225]
        self.preprocessor = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(mean, std)
        ])

    def forward(self, inputs):
        return self.network(inputs).T

    def get_num_heads(self):
        return self.n_heads

    def preprocess(self, batch, device='cpu'):
        tensorize = transforms.ToTensor()
        out = torch.stack([tensorize(img.open()) for img in batch], dim=0).to(device)
        return self.preprocessor(out)

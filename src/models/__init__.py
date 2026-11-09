"""Model zoo for the FedASMU reproduction.

* ``cnn``     : the "synthetic CNN" of the FedASMU paper, arXiv appendix Table 3
                (conv64 k2 -> act -> conv32 k2 -> act -> flatten -> dense).  The
                activation type is not given ("null" parameters) -> ReLU assumed.
* ``vgg11``   : VGG-11 (Simonyan & Zisserman 2015, config A) adapted to 32x32
                inputs; no BatchNorm (the original VGG has none), classifier
                512-512-C with dropout as in common CIFAR FL code (FedMA/FedNova).
* ``lenet5``  : LeCun-style LeNet-5 (conv6-pool-conv16-pool-fc120-fc84-fc).
* ``resnet20``: He et al. 2016 CIFAR ResNet-20 (3x3 stages 16/32/64, BN).
* ``alexnet`` : CIFAR-sized AlexNet variant (5 conv + 3 fc).
* ``textcnn`` : Kim-2014 style TextCNN (trainable embedding, filters of width 3/4/5).
Only ``cnn`` is fully specified by the paper; the others follow the cited
standard architectures (documented in docs/PAPER_REPRODUCTION.md).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperCNN(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, img=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=2)
        self.fc = nn.Linear(32 * (img - 2) * (img - 2), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(torch.flatten(x, 1))


class VGG11(nn.Module):
    CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        layers, c = [], in_ch
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]
                c = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, img=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        s = ((img - 4) // 2 - 4) // 2
        self.fc1 = nn.Linear(16 * s * s, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class _Basic(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, s, 1, bias=False)
        self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(o)
        self.sc = nn.Sequential()
        if s != 1 or i != o:
            self.sc = nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(y)) + self.sc(x))


class ResNet20(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        layers, c = [], 16
        for o, s in [(16, 1), (32, 2), (64, 2)]:
            for k in range(3):
                layers.append(_Basic(c, o, s if k == 0 else 1))
                c = o
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layers(F.relu(self.bn(self.conv(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class AlexNetCifar(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 2 * 2, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


class TextCNN(nn.Module):
    """Embedding 100, 100 filters each of width 3/4/5, max pooling, dropout 0.5 (sizes assumed)."""

    def __init__(self, num_classes=2, vocab_size=20002, embed_dim=100, filters=100, widths=(3, 4, 5)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, filters, w) for w in widths)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(filters * len(widths), num_classes)

    def forward(self, x):
        e = self.embedding(x).transpose(1, 2)
        pooled = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


MODELS = {"cnn": PaperCNN, "vgg11": VGG11, "lenet5": LeNet5, "resnet20": ResNet20, "alexnet": AlexNetCifar,
          "textcnn": TextCNN}
# Constructor arguments for inputs that differ from 3x32x32
DATASET_KWARGS = {("lenet5", "fmnist"): {"in_ch": 1, "img": 28}}


def build_model(name: str, num_classes: int, dataset: str = None) -> nn.Module:
    kwargs = DATASET_KWARGS.get((name.lower(), (dataset or "").lower()), {})
    return MODELS[name.lower()](num_classes=num_classes, **kwargs)

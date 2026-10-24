import torch
from torchvision.ops.stochastic_depth import StochasticDepth

class ConvNet(torch.nn.Module):
    """A mostly faithful recreation of LeNet 5."""

    def __init__(self, nonlinearity = torch.nn.ReLU, classes=10):
        super(ConvNet, self).__init__()
        self.net = torch.nn.Sequential(
                # Basic ConvNet as in the 20xx years
                torch.nn.Conv2d(in_channels=1, out_channels=15, kernel_size=(3,3), padding=1, stride=2),
                nonlinearity(),
                torch.nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3,3), padding=1, stride=2),
                nonlinearity(),
                torch.nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3,3), padding=1, stride=2),
                nonlinearity(),
                torch.nn.Flatten(),
                torch.nn.Linear(240, 60),
                torch.nn.Linear(60, classes),
                )
        self.decision = torch.nn.Softmax(dim=1)

        for layer in [0, 2, 4, 7]:
            torch.nn.init.kaiming_normal_(self.net[layer].weight.data, nonlinearity="relu")

    def forward(self, x):
        """Forward through the network."""
        y_hat = self.decision(self.net(x))
        return y_hat

class WiderConvNet(torch.nn.Module):
    """A wider recreation of LeNet 5."""

    def __init__(self, nonlinearity = torch.nn.ReLU, classes=10):
        super(WiderConvNet, self).__init__()
        self.net = torch.nn.Sequential(
                # Basic ConvNet as in the 20xx years
                torch.nn.Conv2d(in_channels=1, out_channels=15, kernel_size=(3,3), padding=1, stride=2),
                nonlinearity(),
                torch.nn.Conv2d(in_channels=15, out_channels=30, kernel_size=(3,3), padding=1, stride=2),
                nonlinearity(),
                torch.nn.Conv2d(in_channels=30, out_channels=60, kernel_size=(3,3), padding=1, stride=2),
                nonlinearity(),
                torch.nn.Flatten(),
                torch.nn.Linear(4*240, 60),
                torch.nn.Linear(60, classes),
                )
        self.decision = torch.nn.Softmax(dim=1)

        for layer in [0, 2, 4, 7]:
            torch.nn.init.kaiming_normal_(self.net[layer].weight.data, nonlinearity="relu")

    def forward(self, x):
        """Forward through the network."""
        y_hat = self.decision(self.net(x))
        return y_hat

class DeeperConvNet(torch.nn.Module):
    """A deeper recreation of LeNet 5."""

    def __init__(self, nonlinearity = torch.nn.ReLU, classes=10):
        super(DeeperConvNet, self).__init__()
        self.net = torch.nn.Sequential(
                # Basic ConvNet as in the 20xx years
                torch.nn.Conv2d(in_channels=1, out_channels=15, kernel_size=(3,3), padding=1, stride=1),
                nonlinearity(),
                torch.nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3,3), padding=1, stride=2),
                nonlinearity(),
                torch.nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3,3), padding=1, stride=1),
                nonlinearity(),
                torch.nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3,3), padding=1, stride=2),
                nonlinearity(),
                torch.nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3,3), padding=1, stride=2),
                nonlinearity(),
                torch.nn.Flatten(),
                torch.nn.Linear(240, 60),
                torch.nn.Linear(60, classes),
                )
        self.decision = torch.nn.Softmax(dim=1)

        for layer in [0, 2, 4, 6, 8, 11]:
            torch.nn.init.kaiming_normal_(self.net[layer].weight.data, nonlinearity="relu")

    def forward(self, x):
        """Forward through the network."""
        y_hat = self.decision(self.net(x))
        return y_hat


class LayerNormChannelsFirst(torch.nn.Module):
    """LayerNorm working on (N, C, H, W) tensors directly."""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        # Normalize across channel dimension
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(torch.nn.Module):
    """ConvNeXt Block with 7x7 Depthwise Conv, Inverted Bottleneck, GELU, and LayerNorm."""
    def __init__(self, dim, layer_scale_init_value=1e-6, stoch_depth=None):
        super().__init__()
        # 1. Depthwise 7x7 Conv (Large kernel, spatially decoupled)
        self.dwconv = torch.nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNormChannelsFirst(dim)

        # 2. Inverted Bottleneck: dim -> 4*dim -> dim
        self.pwconv1 = torch.nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Conv2d(4 * dim, dim, kernel_size=1)

        # Layer Scale for training stability
        self.gamma = torch.nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

        if stoch_depth is None:
            self.stoch_depth = None
        else:
            self.stoch_depth = StochasticDepth(stoch_depth, "row")

    def forward(self, x):
        y = self.dwconv(x)
        y = self.norm(y)
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.pwconv2(y)

        if self.gamma is not None:
            y = self.gamma[:, None, None] * y

        if self.stoch_depth is not None:
            y = self.stoch_depth(y)

        return x + y

class ConvNeXtEMNIST(torch.nn.Module):
    def __init__(self, classes=10, depths=[2, 2, 2], dims=[16, 32, 64], stoch_depth=True):
        super().__init__()

        # The stem uses a smaller convolution than for something like ImageNet
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=dims[0], kernel_size=3, padding=1),
            LayerNormChannelsFirst(dims[0])
        )

        # Downsample layers between stages (stride 2)
        self.downsample_layers = torch.nn.ModuleList()
        self.downsample_layers.append(self.stem)
        for i in range(len(dims)-1):
            self.downsample_layers.append(torch.nn.Sequential(
                LayerNormChannelsFirst(dims[i]),
                torch.nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            )

        # Calculate depths for stochastic depth
        total_depth = sum(depths)
        cur_depth = 0
        # Stages containing ConvNeXt blocks
        self.stages = torch.nn.ModuleList()
        for i in range(len(dims)):
            if stoch_depth is not None:
                stage = torch.nn.Sequential(
                    *[ConvNeXtBlock(dim=dims[i], stoch_depth=0.4*(cur_depth+j)/total_depth) for j in range(depths[i])]
                )
                cur_depth = cur_depth + depths[i]
            else:
                stage = torch.nn.Sequential(
                    *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
                )
            self.stages.append(stage)

        # Classification Head
        self.norm = LayerNormChannelsFirst(dims[-1])
        self.head = torch.nn.Linear(dims[-1], classes)

    def forward(self, x):
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.norm(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

class Linear(torch.nn.Module):
    """A linear neural network."""

    def __init__(self, nonlinearity = torch.nn.ReLU, classes=10):
        super(Linear, self).__init__()
        self.net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(28*28, 512),
                nonlinearity(),
                torch.nn.Linear(512, 120),
                nonlinearity(),
                torch.nn.Linear(120, 84),
                torch.nn.Linear(84, classes)
                )
        self.decision = torch.nn.Softmax(dim=1)

        # Proper initialization makes it more likely to find a good gradient to follow
        torch.nn.init.kaiming_normal_(self.net[1].weight.data, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.net[3].weight.data, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.net[5].weight.data, nonlinearity="relu")

    def forward(self, x):
        """Forward through the network."""
        y_hat = self.decision(self.net(x))
        return y_hat



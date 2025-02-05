import torch
import torch.nn as nn
from swin_stem import SwinStemTransformer


class AerialFormer(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Initialize SwinStemTransformer backbone
        self.backbone = SwinStemTransformer(
            pretrain_img_size=512,
        )

        # Initialize MDCDecoder
        self.decoder = MDCDecoder(
            in_channels=[48, 96, 192, 384, 768],  # Matches backbone outputs
            channels=96,
            num_classes=num_classes,
        )

    def forward(self, x):
        # Backbone returns features from 5 stages
        features = self.backbone(x)

        # Verify feature dimensions
        assert len(features) == 5, f"Expected 5 features, got {len(features)}"

        return self.decoder(features)


class MDCDecoder(nn.Module):
    def __init__(self, in_channels, channels=96, num_classes=7):
        super().__init__()
        self.in_channels = list(reversed(in_channels))

        # Upsampling modules
        self.up_convs = nn.ModuleList([nn.Identity()])  # First stage
        self.dilated_convs = nn.ModuleList()

        # Configuration for multi-dilated convolutions
        custom_params_list = [
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
            {"kernel": (3, 3, 3), "padding": (1, 1, 1), "dilation": (1, 1, 1)},
            {"kernel": (1, 3, 3), "padding": (0, 1, 1), "dilation": (1, 1, 1)},
        ]

        # Create processing blocks for each stage
        for idx in range(1, len(self.in_channels)):
            # Upsampling path
            self.up_convs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.in_channels[idx - 1],
                        self.in_channels[idx],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.in_channels[idx]),
                    nn.ReLU6(),
                )
            )

            # Feature processing
            self.dilated_convs.append(
                nn.Sequential(
                    MDCBlock(
                        in_channels=self.in_channels[idx] * 2,  # After concatenation
                        out_channels=self.in_channels[idx],
                        custom_params=custom_params_list[idx],
                    ),
                    ConvBNReLU(self.in_channels[idx], self.in_channels[idx]),
                )
            )

        # First stage processing
        self.dilated_convs.insert(
            0,
            nn.Sequential(
                MDCBlock(
                    self.in_channels[0], self.in_channels[0], custom_params_list[0]
                ),
                ConvBNReLU(self.in_channels[0], self.in_channels[0]),
            ),
        )

        # Final prediction
        self.final_conv = nn.Conv2d(self.in_channels[-1], num_classes, kernel_size=1)

    def forward(self, features):
        features = list(reversed(features))
        x = features[0]
        x = self.dilated_convs[0](x)

        for idx in range(1, len(self.in_channels)):
            x = self.up_convs[idx](x)
            x = torch.cat([x, features[idx]], dim=1)
            x = self.dilated_convs[idx](x)

        return self.final_conv(x)


# Helper classes
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(dilation * (kernel_size - 1)) // 2,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )


class MDCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, custom_params):
        super().__init__()
        self.split_num = 3
        self.pre_conv = ConvBNReLU(in_channels, in_channels, kernel_size=1)

        # Channel splitting
        quotient, reminder = divmod(in_channels, self.split_num)
        self.split_channels = [quotient + (1 if i < reminder else 0) for i in range(3)]

        # Parallel convolutions
        self.conv_layers = nn.ModuleList()
        for i in range(3):
            self.conv_layers.append(
                nn.Conv2d(
                    self.split_channels[i],
                    self.split_channels[i],
                    kernel_size=custom_params["kernel"][i],
                    padding=custom_params["padding"][i],
                    dilation=custom_params["dilation"][i],
                    bias=False,
                )
            )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, x):
        x = self.pre_conv(x)
        split_x = torch.split(x, self.split_channels, dim=1)
        processed = [conv(part) for conv, part in zip(self.conv_layers, split_x)]
        x = torch.cat(processed, dim=1)
        return self.fusion(x)

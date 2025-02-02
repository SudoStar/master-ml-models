import torch
import torch.nn as nn


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

        # Calculate split channels
        quotient, reminder = divmod(in_channels, self.split_num)
        self.split_channels = [
            quotient + (1 if i < reminder else 0) for i in range(self.split_num)
        ]

        # Create convolution layers for each split
        self.conv_layers = nn.ModuleList()
        for i in range(self.split_num):
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

        # Fusion layer with proper channel alignment
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, x):
        x = self.pre_conv(x)
        split_x = torch.split(x, self.split_channels, dim=1)
        processed = []
        for conv, x_part in zip(self.conv_layers, split_x):
            processed.append(conv(x_part))
        x = torch.cat(processed, dim=1)
        return self.fusion(x)


class MDCDecoder(nn.Module):
    def __init__(self, in_channels, channels=96, num_classes=7):
        super().__init__()
        self.in_channels = list(reversed(in_channels))

        # Verify input channels match the backbone outputs
        print(
            f"MDCDecoder expects {len(self.in_channels)} input features with channels: {self.in_channels}"
        )

        self.up_convs = nn.ModuleList([nn.Identity()])  # First element is Identity
        self.dilated_convs = nn.ModuleList()

        # Configuration for multi-dilated convolutions
        custom_params_list = [
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
            {"kernel": (3, 3, 3), "padding": (1, 1, 1), "dilation": (1, 1, 1)},
            {"kernel": (1, 3, 3), "padding": (0, 1, 1), "dilation": (1, 1, 1)},
        ]

        # Create upsampling and processing blocks
        for idx in range(1, len(self.in_channels)):
            # Upsampling layers
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

            # Dilated convolution blocks
            self.dilated_convs.append(
                nn.Sequential(
                    MDCBlock(
                        self.in_channels[idx] * 2,  # After concatenation
                        self.in_channels[idx],
                        custom_params_list[idx],
                    ),
                    ConvBNReLU(self.in_channels[idx], self.in_channels[idx]),
                )
            )

        # First layer processing without upsampling
        self.dilated_convs.insert(
            0,
            nn.Sequential(
                MDCBlock(
                    self.in_channels[0], self.in_channels[0], custom_params_list[0]
                ),
                ConvBNReLU(self.in_channels[0], self.in_channels[0]),
            ),
        )

        # Final prediction layer
        self.final_conv = nn.Conv2d(self.in_channels[-1], num_classes, kernel_size=1)

    def forward(self, features):
        features = list(reversed(features))
        x = features[0]

        # Process first feature
        x = self.dilated_convs[0](x)

        # Process subsequent features
        for idx in range(1, len(self.in_channels)):
            x = self.up_convs[idx](x)
            x = torch.cat([x, features[idx]], dim=1)
            x = self.dilated_convs[idx](x)

        return self.final_conv(x)


class AerialFormer(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Initialize your actual backbone (SwinStemTransformer)
        self.backbone = timm.create_model(
            "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
            features_only=True,
            output_stride=32,
            img_size=512,
            out_indices=(0, 1, 2, 3, 4),  # Get all 5 stages
            pretrained=True,
        )

        # MDCDecoder with correct channel specification
        self.decoder = MDCDecoder(
            in_channels=self.backbone.feature_info.channels(),  # Must match backbone outputs
            channels=96,
            num_classes=num_classes,
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.decoder(features)

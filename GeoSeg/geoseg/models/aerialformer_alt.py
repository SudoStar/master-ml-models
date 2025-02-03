import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MDCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super().__init__()
        norm_cfg = norm_cfg or dict(type="BN")
        act_cfg = act_cfg or dict(type="ReLU")

        self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Split input into 3 parts
        self.layers = nn.ModuleList()
        quotient = in_channels // 3
        reminder = in_channels % 3

        split_channels = [quotient] * 3
        if reminder == 1:
            split_channels[0] += 1
        elif reminder == 2:
            split_channels[0] += 1
            split_channels[1] += 1

        # Custom dilated convolutions with different kernel sizes and dilations
        custom_params = [
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
        ]

        for kernel, padding, dilation, channels in zip(
            custom_params[0]["kernel"],
            custom_params[0]["padding"],
            custom_params[0]["dilation"],
            split_channels,
        ):
            self.layers.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )

        self.fusion_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pre_conv(x)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)

        x1 = self.layers[0](x1)
        x2 = self.layers[1](x2)
        x3 = self.layers[2](x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fusion_layer(x)

        return self.act(self.norm(x))


class Decoder(nn.Module):
    def __init__(
        self, encoder_channels=(48, 96, 192, 384), decoder_channels=96, num_classes=7
    ):
        super().__init__()

        # Reversed channels for top-down approach
        self.in_channels = list(reversed(encoder_channels))

        # Up-convolution and dilated convolution layers
        self.up_convs = nn.ModuleList()
        self.dilated_convs = nn.ModuleList()

        for idx in range(len(self.in_channels)):
            # Up-sampling convolution
            if idx != 0:
                self.up_convs.append(
                    nn.ConvTranspose2d(
                        self.in_channels[idx - 1],
                        self.in_channels[idx],
                        kernel_size=2,
                        stride=2,
                    )
                )
            else:
                self.up_convs.append(nn.Identity())

            # Multi-dilated convolution block
            self.dilated_convs.append(
                nn.Sequential(
                    MDCBlock(
                        in_channels=self.in_channels[idx] * (2 if idx != 0 else 1),
                        out_channels=self.in_channels[idx],
                    ),
                    ConvBNReLU(
                        self.in_channels[idx],
                        self.in_channels[idx],
                        kernel_size=3,
                        padding=1,
                    ),
                )
            )

        # Final classification layer
        self.cls_seg = nn.Conv2d(self.in_channels[-1], num_classes, kernel_size=1)

    def forward(self, inputs):
        # Reverse and transform inputs
        inputs = list(reversed(inputs))

        x = inputs[0]
        x = self.dilated_convs[0](x)

        for idx in range(1, len(inputs)):
            x = self.up_convs[idx](x)
            x = torch.cat([x, inputs[idx]], dim=1)
            x = self.dilated_convs[idx](x)

        return self.cls_seg(x)


class AerialFormer(nn.Module):
    def __init__(
        self,
        backbone_name="swin_base_patch4_window12_384.ms_in22k_ft_in1k",
        num_classes=7,
        pretrained=True,
    ):
        super().__init__()

        # Use timm for backbone to simplify loading
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            img_size=512,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
        )

        # Decoder
        self.decoder = Decoder(
            encoder_channels=self.backbone.feature_info.channels(),
            num_classes=num_classes,
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.decoder(features)

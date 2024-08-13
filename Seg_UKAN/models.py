import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models import DenseNet
from torchvision.models.densenet import _Transition, _load_state_dict
from collections import OrderedDict
from building_blocks import (
    ConvSODEFunc,
    ConvSODEFunc3D,
    ConvODEFunc,
    ConvResFunc,
    InitialVelocity,
    InitialVelocity3D,
    ODEBlock,
    LevelBlock,
    LevelBlock3D,
    get_nonlinearity,
)

# Second-order ODE UNet
class ConvSODEUNet(nn.Module):
    def __init__(
        self,
        num_filters,
        output_dim=1,
        time_dependent=False,
        non_linearity="softplus",
        tol=1e-3,
        adjoint=False,
        method="rk4"
    ):
        """
        ConvSODEUNet (Second order ODE UNet)
        Args:
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
            tol (float): tolerance to be used for ODE solver
            adjoint (bool): whether to use the adjoint method to calculate the gradients
        """
        super(ConvSODEUNet, self).__init__()
        nf = num_filters
        self.method = method
        print(f"Solver: {method}")

        self.input_1x1 = nn.Conv2d(3, nf, 1, 1)
        self.initial_velocity = InitialVelocity(nf, non_linearity)

        ode_down1 = ConvSODEFunc(nf * 2, time_dependent, non_linearity)
        self.odeblock_down1 = ODEBlock(ode_down1, tol=tol, adjoint=adjoint)
        self.conv_down1_2 = nn.Conv2d(nf * 2, nf * 4, 1, 1)

        ode_down2 = ConvSODEFunc(nf * 4, time_dependent, non_linearity)
        self.odeblock_down2 = ODEBlock(ode_down2, tol=tol, adjoint=adjoint)
        self.conv_down2_3 = nn.Conv2d(nf * 4, nf * 8, 1, 1)

        ode_down3 = ConvSODEFunc(nf * 8, time_dependent, non_linearity)
        self.odeblock_down3 = ODEBlock(ode_down3, tol=tol, adjoint=adjoint)
        self.conv_down3_4 = nn.Conv2d(nf * 8, nf * 16, 1, 1)

        ode_down4 = ConvSODEFunc(nf * 16, time_dependent, non_linearity)
        self.odeblock_down4 = ODEBlock(ode_down4, tol=tol, adjoint=adjoint)
        self.conv_down4_embed = nn.Conv2d(nf * 16, nf * 32, 1, 1)

        ode_embed = ConvSODEFunc(nf * 32, time_dependent, non_linearity)
        self.odeblock_embedding = ODEBlock(ode_embed, tol=tol, adjoint=adjoint)
        self.conv_up_embed_1 = nn.Conv2d(nf * 32 + nf * 16, nf * 16, 1, 1)

        ode_up1 = ConvSODEFunc(nf * 16, time_dependent, non_linearity)
        self.odeblock_up1 = ODEBlock(ode_up1, tol=tol, adjoint=adjoint)
        self.conv_up1_2 = nn.Conv2d(nf * 16 + nf * 8, nf * 8, 1, 1)

        ode_up2 = ConvSODEFunc(nf * 8, time_dependent, non_linearity)
        self.odeblock_up2 = ODEBlock(ode_up2, tol=tol, adjoint=adjoint)
        self.conv_up2_3 = nn.Conv2d(nf * 8 + nf * 4, nf * 4, 1, 1)

        ode_up3 = ConvSODEFunc(nf * 4, time_dependent, non_linearity)
        self.odeblock_up3 = ODEBlock(ode_up3, tol=tol, adjoint=adjoint)
        self.conv_up3_4 = nn.Conv2d(nf * 4 + nf * 2, nf * 2, 1, 1)

        ode_up4 = ConvSODEFunc(nf * 2, time_dependent, non_linearity)
        self.odeblock_up4 = ODEBlock(ode_up4, tol=tol, adjoint=adjoint)

        self.classifier = nn.Conv2d(nf * 2, output_dim, 1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        x = self.initial_velocity(x)

        features1 = self.odeblock_down1(x,  method=self.method)  # 512
        x = self.non_linearity(self.conv_down1_2(features1))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features2 = self.odeblock_down2(x,  method=self.method)  # 256
        x = self.non_linearity(self.conv_down2_3(features2))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features3 = self.odeblock_down3(x,  method=self.method)  # 128
        x = self.non_linearity(self.conv_down3_4(features3))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features4 = self.odeblock_down4(x,  method=self.method)  # 64
        x = self.non_linearity(self.conv_down4_embed(features4))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        x = self.odeblock_embedding(x,  method=self.method)  # 32

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features4), dim=1)
        x = self.non_linearity(self.conv_up_embed_1(x))
        x = self.odeblock_up1(x,  method=self.method)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features3), dim=1)
        x = self.non_linearity(self.conv_up1_2(x))
        x = self.odeblock_up2(x,  method=self.method)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features2), dim=1)
        x = self.non_linearity(self.conv_up2_3(x))
        x = self.odeblock_up3(x,  method=self.method)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features1), dim=1)
        x = self.non_linearity(self.conv_up3_4(x))
        x = self.odeblock_up4(x,  method=self.method)

        pred = self.classifier(x)
        return pred

class ConvSODEUNet3D(nn.Module):
    def __init__(
        self,
        num_filters,
        output_dim=1,
        time_dependent=False,
        non_linearity="softplus",
        tol=1e-3,
        adjoint=False,
        method="rk4"
    ):
        """
        ConvSODEUNet (Second order ODE UNet)
        Args:
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
            tol (float): tolerance to be used for ODE solver
            adjoint (bool): whether to use the adjoint method to calculate the gradients
        """
        super(ConvSODEUNet3D, self).__init__()
        nf = num_filters
        self.method = method
        print(f"Solver: {method}")

        self.input_1x1 = nn.Conv3d(1, nf, 1, 1)
        self.initial_velocity = InitialVelocity3D(nf, non_linearity)

        ode_down1 = ConvSODEFunc3D(nf * 2, time_dependent, non_linearity)
        self.odeblock_down1 = ODEBlock(ode_down1, tol=tol, adjoint=adjoint)
        self.conv_down1_2 = nn.Conv3d(nf * 2, nf * 4, 1, 1)

        ode_down2 = ConvSODEFunc3D(nf * 4, time_dependent, non_linearity)
        self.odeblock_down2 = ODEBlock(ode_down2, tol=tol, adjoint=adjoint)
        self.conv_down2_3 = nn.Conv3d(nf * 4, nf * 8, 1, 1)

        ode_down3 = ConvSODEFunc3D(nf * 8, time_dependent, non_linearity)
        self.odeblock_down3 = ODEBlock(ode_down3, tol=tol, adjoint=adjoint)
        self.conv_down3_4 = nn.Conv3d(nf * 8, nf * 16, 1, 1)

        ode_down4 = ConvSODEFunc3D(nf * 16, time_dependent, non_linearity)
        self.odeblock_down4 = ODEBlock(ode_down4, tol=tol, adjoint=adjoint)
        self.conv_down4_embed = nn.Conv3d(nf * 16, nf * 32, 1, 1)

        ode_embed = ConvSODEFunc3D(nf * 32, time_dependent, non_linearity)
        self.odeblock_embedding = ODEBlock(ode_embed, tol=tol, adjoint=adjoint)
        self.conv_up_embed_1 = nn.Conv3d(nf * 32 + nf * 16, nf * 16, 1, 1)

        ode_up1 = ConvSODEFunc3D(nf * 16, time_dependent, non_linearity)
        self.odeblock_up1 = ODEBlock(ode_up1, tol=tol, adjoint=adjoint)
        self.conv_up1_2 = nn.Conv3d(nf * 16 + nf * 8, nf * 8, 1, 1)

        ode_up2 = ConvSODEFunc3D(nf * 8, time_dependent, non_linearity)
        self.odeblock_up2 = ODEBlock(ode_up2, tol=tol, adjoint=adjoint)
        self.conv_up2_3 = nn.Conv3d(nf * 8 + nf * 4, nf * 4, 1, 1)

        ode_up3 = ConvSODEFunc3D(nf * 4, time_dependent, non_linearity)
        self.odeblock_up3 = ODEBlock(ode_up3, tol=tol, adjoint=adjoint)
        self.conv_up3_4 = nn.Conv3d(nf * 4 + nf * 2, nf * 2, 1, 1)

        ode_up4 = ConvSODEFunc3D(nf * 2, time_dependent, non_linearity)
        self.odeblock_up4 = ODEBlock(ode_up4, tol=tol, adjoint=adjoint)

        self.classifier = nn.Conv3d(nf * 2, output_dim, 1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        x = self.input_1x1(x)
        x = self.initial_velocity(x)

        features1 = self.odeblock_down1(x,  method=self.method)  # 512
        x = self.non_linearity(self.conv_down1_2(features1))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="trilinear", align_corners=False
        )

        features2 = self.odeblock_down2(x,  method=self.method)  # 256
        x = self.non_linearity(self.conv_down2_3(features2))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="trilinear", align_corners=False
        )

        features3 = self.odeblock_down3(x,  method=self.method)  # 128
        x = self.non_linearity(self.conv_down3_4(features3))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="trilinear", align_corners=False
        )

        features4 = self.odeblock_down4(x,  method=self.method)  # 64
        x = self.non_linearity(self.conv_down4_embed(features4))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="trilinear", align_corners=False
        )

        x = self.odeblock_embedding(x,  method=self.method)  # 32

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="trilinear", align_corners=False
        )
        x = torch.cat((x, features4), dim=1)
        x = self.non_linearity(self.conv_up_embed_1(x))
        x = self.odeblock_up1(x,  method=self.method)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="trilinear", align_corners=False
        )
        x = torch.cat((x, features3), dim=1)
        x = self.non_linearity(self.conv_up1_2(x))
        x = self.odeblock_up2(x,  method=self.method)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="trilinear", align_corners=False
        )
        x = torch.cat((x, features2), dim=1)
        x = self.non_linearity(self.conv_up2_3(x))
        x = self.odeblock_up3(x,  method=self.method)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="trilinear", align_corners=False
        )
        x = torch.cat((x, features1), dim=1)
        x = self.non_linearity(self.conv_up3_4(x))
        x = self.odeblock_up4(x,  method=self.method)

        pred = self.classifier(x)
        return pred    

class ConvODEUNet(nn.Module):
    def __init__(
        self,
        num_filters,
        output_dim=1,
        time_dependent=False,
        non_linearity="softplus",
        tol=1e-3,
        adjoint=False,
    ):
        """
        ConvODEUNet (U-Node in paper)
        Args:
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
            tol (float): tolerance to be used for ODE solver
            adjoint (bool): whether to use the adjoint method to calculate the gradients
        """
        super(ConvODEUNet, self).__init__()
        nf = num_filters

        self.input_1x1 = nn.Conv2d(3, nf, 1, 1)

        ode_down1 = ConvODEFunc(nf, time_dependent, non_linearity)
        self.odeblock_down1 = ODEBlock(ode_down1, tol=tol, adjoint=adjoint)
        self.conv_down1_2 = nn.Conv2d(nf, nf * 2, 1, 1)

        ode_down2 = ConvODEFunc(nf * 2, time_dependent, non_linearity)
        self.odeblock_down2 = ODEBlock(ode_down2, tol=tol, adjoint=adjoint)
        self.conv_down2_3 = nn.Conv2d(nf * 2, nf * 4, 1, 1)

        ode_down3 = ConvODEFunc(nf * 4, time_dependent, non_linearity)
        self.odeblock_down3 = ODEBlock(ode_down3, tol=tol, adjoint=adjoint)
        self.conv_down3_4 = nn.Conv2d(nf * 4, nf * 8, 1, 1)

        ode_down4 = ConvODEFunc(nf * 8, time_dependent, non_linearity)
        self.odeblock_down4 = ODEBlock(ode_down4, tol=tol, adjoint=adjoint)
        self.conv_down4_embed = nn.Conv2d(nf * 8, nf * 16, 1, 1)

        ode_embed = ConvODEFunc(nf * 16, time_dependent, non_linearity)
        self.odeblock_embedding = ODEBlock(ode_embed, tol=tol, adjoint=adjoint)

        self.conv_up_embed_1 = nn.Conv2d(nf * 16 + nf * 8, nf * 8, 1, 1)
        ode_up1 = ConvODEFunc(nf * 8, time_dependent, non_linearity)
        self.odeblock_up1 = ODEBlock(ode_up1, tol=tol, adjoint=adjoint)

        self.conv_up1_2 = nn.Conv2d(nf * 8 + nf * 4, nf * 4, 1, 1)
        ode_up2 = ConvODEFunc(nf * 4, time_dependent, non_linearity)
        self.odeblock_up2 = ODEBlock(ode_up2, tol=tol, adjoint=adjoint)

        self.conv_up2_3 = nn.Conv2d(nf * 4 + nf * 2, nf * 2, 1, 1)
        ode_up3 = ConvODEFunc(nf * 2, time_dependent, non_linearity)
        self.odeblock_up3 = ODEBlock(ode_up3, tol=tol, adjoint=adjoint)

        self.conv_up3_4 = nn.Conv2d(nf * 2 + nf, nf, 1, 1)
        ode_up4 = ConvODEFunc(nf, time_dependent, non_linearity)
        self.odeblock_up4 = ODEBlock(ode_up4, tol=tol, adjoint=adjoint)

        self.classifier = nn.Conv2d(nf, output_dim, 1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x, return_features=False):
        x = self.non_linearity(self.input_1x1(x))

        features1 = self.odeblock_down1(x)  # 512
        x = self.non_linearity(self.conv_down1_2(features1))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features2 = self.odeblock_down2(x)  # 256
        x = self.non_linearity(self.conv_down2_3(features2))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features3 = self.odeblock_down3(x)  # 128
        x = self.non_linearity(self.conv_down3_4(features3))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features4 = self.odeblock_down4(x)  # 64
        x = self.non_linearity(self.conv_down4_embed(features4))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        x = self.odeblock_embedding(x)  # 32

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features4), dim=1)
        x = self.non_linearity(self.conv_up_embed_1(x))
        x = self.odeblock_up1(x)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features3), dim=1)
        x = self.non_linearity(self.conv_up1_2(x))
        x = self.odeblock_up2(x)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features2), dim=1)
        x = self.non_linearity(self.conv_up2_3(x))
        x = self.odeblock_up3(x)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features1), dim=1)
        x = self.non_linearity(self.conv_up3_4(x))
        x = self.odeblock_up4(x)

        pred = self.classifier(x)
        return pred


class ConvResUNet(nn.Module):
    def __init__(self, num_filters, output_dim=1, non_linearity="softplus"):
        """
        ConvResUNet (U-Node in paper)
        Args:
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvResUNet, self).__init__()
        self.output_dim = output_dim

        self.input_1x1 = nn.Conv2d(3, num_filters, 1, 1)

        self.block_down1 = ConvResFunc(num_filters, non_linearity)
        self.conv_down1_2 = nn.Conv2d(num_filters, num_filters * 2, 1, 1)
        self.block_down2 = ConvResFunc(num_filters * 2, non_linearity)
        self.conv_down2_3 = nn.Conv2d(num_filters * 2, num_filters * 4, 1, 1)
        self.block_down3 = ConvResFunc(num_filters * 4, non_linearity)
        self.conv_down3_4 = nn.Conv2d(num_filters * 4, num_filters * 8, 1, 1)
        self.block_down4 = ConvResFunc(num_filters * 8, non_linearity)
        self.conv_down4_embed = nn.Conv2d(num_filters * 8, num_filters * 16, 1, 1)

        self.block_embedding = ConvResFunc(num_filters * 16, non_linearity)

        self.conv_up_embed_1 = nn.Conv2d(
            num_filters * 16 + num_filters * 8, num_filters * 8, 1, 1
        )
        self.block_up1 = ConvResFunc(num_filters * 8, non_linearity)
        self.conv_up1_2 = nn.Conv2d(
            num_filters * 8 + num_filters * 4, num_filters * 4, 1, 1
        )
        self.block_up2 = ConvResFunc(num_filters * 4, non_linearity)
        self.conv_up2_3 = nn.Conv2d(
            num_filters * 4 + num_filters * 2, num_filters * 2, 1, 1
        )
        self.block_up3 = ConvResFunc(num_filters * 2, non_linearity)
        self.conv_up3_4 = nn.Conv2d(num_filters * 2 + num_filters, num_filters, 1, 1)
        self.block_up4 = ConvResFunc(num_filters, non_linearity)

        self.classifier = nn.Conv2d(num_filters, self.output_dim, 1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x, return_features=False):
        x = self.non_linearity(self.input_1x1(x))

        features1 = self.block_down1(x)  # 512
        x = self.non_linearity(self.conv_down1_2(x))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features2 = self.block_down2(x)  # 256
        x = self.non_linearity(self.conv_down2_3(x))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features3 = self.block_down3(x)  # 128
        x = self.non_linearity(self.conv_down3_4(x))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        features4 = self.block_down4(x)  # 64
        x = self.non_linearity(self.conv_down4_embed(x))
        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        x = self.block_embedding(x)  # 32

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features4), dim=1)
        x = self.non_linearity(self.conv_up_embed_1(x))
        x = self.block_up1(x)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features3), dim=1)
        x = self.non_linearity(self.conv_up1_2(x))
        x = self.block_up2(x)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features2), dim=1)
        x = self.non_linearity(self.conv_up2_3(x))
        x = self.block_up3(x)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat((x, features1), dim=1)
        x = self.non_linearity(self.conv_up3_4(x))
        x = self.block_up4(x)

        pred = self.classifier(x)
        return pred


class Unet(nn.Module):
    def __init__(self, depth, num_filters, output_dim, block):
        """
        Unet
        Args:
            depth (int): number of levels of UNet
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
        """
        super(Unet, self).__init__()
        self.main = LevelBlock(depth, depth, 3, num_filters, block=block)
        main_out = list(self.main.modules())[-2].out_channels
        self.out = nn.Conv2d(main_out, output_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        x = self.main(inp)
        return self.out(x)

class Unet3D(nn.Module):
    def __init__(self, depth, num_filters, output_dim, block):
        """
        Unet
        Args:
            depth (int): number of levels of UNet
            num_filters (int): number of filters for first conv layer
            output_dim (int): how many feature maps the network outputs
        """
        super(Unet3D, self).__init__()
        self.main = LevelBlock3D(depth, depth, 1, num_filters, block=block)
        main_out = list(self.main.modules())[-2].out_channels
        self.out = nn.Conv3d(main_out, output_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        x = self.main(inp)
        return self.out(x)


"""
Code copied from https://github.com/jeya-maria-jose/UNeXt-pytorch/blob/main/archs.py 
"""


class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(
        self,
        num_classes,
        input_channels=3,
        deep_supervision=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dims=[128, 160, 256],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[1, 1, 1],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    ):
        super().__init__()

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList(
            [
                shiftedBlock(
                    dim=embed_dims[1],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.block2 = nn.ModuleList(
            [
                shiftedBlock(
                    dim=embed_dims[2],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.dblock1 = nn.ModuleList(
            [
                shiftedBlock(
                    dim=embed_dims[1],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.dblock2 = nn.ModuleList(
            [
                shiftedBlock(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(
            F.interpolate(
                self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(
            F.interpolate(
                self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(
            F.interpolate(
                self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )
        out = torch.add(out, t2)
        out = F.relu(
            F.interpolate(
                self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode="bilinear"
            )
        )
        out = torch.add(out, t1)
        out = F.relu(
            F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode="bilinear")
        )

        return self.final(out)


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class shiftedBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class shiftmlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        shift_size=5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [
            torch.roll(x_c, shift, 2)
            for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))
        ]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [
            torch.roll(x_c, shift, 3)
            for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))
        ]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


"""
Code copied from https://github.com/stefano-malacrino/DenseUNet-pytorch/blob/master/dense_unet.py 
"""


class _DenseUNetEncoder(DenseNet):
    def __init__(
        self,
        skip_connections,
        growth_rate,
        block_config,
        num_init_features,
        bn_size,
        drop_rate,
        downsample,
    ):
        super(_DenseUNetEncoder, self).__init__(
            growth_rate, block_config, num_init_features, bn_size, drop_rate
        )

        self.skip_connections = skip_connections

        # remove last norm, classifier
        features = OrderedDict(list(self.features.named_children())[:-1])
        delattr(self, "classifier")
        if not downsample:
            features["conv0"].stride = 1
            del features["pool0"]
        self.features = nn.Sequential(features)

        for module in self.features.modules():
            if isinstance(module, nn.AvgPool2d):
                module.register_forward_hook(
                    lambda _, input, output: self.skip_connections.append(input[0])
                )

    def forward(self, x):
        return self.features(x)


class _DenseUNetDecoder(DenseNet):
    def __init__(
        self,
        skip_connections,
        growth_rate,
        block_config,
        num_init_features,
        bn_size,
        drop_rate,
        upsample,
    ):
        super(_DenseUNetDecoder, self).__init__(
            growth_rate, block_config, num_init_features, bn_size, drop_rate
        )

        self.skip_connections = skip_connections
        self.upsample = upsample

        # remove conv0, norm0, relu0, pool0, last denseblock, last norm, classifier
        features = list(self.features.named_children())[4:-2]
        delattr(self, "classifier")

        num_features = num_init_features
        num_features_list = []
        for i, num_layers in enumerate(block_config):
            num_input_features = num_features + num_layers * growth_rate
            num_output_features = num_features // 2
            num_features_list.append((num_input_features, num_output_features))
            num_features = num_input_features // 2

        for i in range(len(features)):
            name, module = features[i]
            if isinstance(module, _Transition):
                num_input_features, num_output_features = num_features_list.pop(1)
                features[i] = (
                    name,
                    _TransitionUp(
                        num_input_features, num_output_features, skip_connections
                    ),
                )

        features.reverse()

        self.features = nn.Sequential(OrderedDict(features))

        num_input_features, _ = num_features_list.pop(0)

        if upsample:
            self.features.add_module(
                "upsample0", nn.Upsample(scale_factor=4, mode="bilinear")
            )
        self.features.add_module("norm0", nn.BatchNorm2d(num_input_features))
        self.features.add_module("relu0", nn.ReLU(inplace=True))
        self.features.add_module(
            "conv0",
            nn.Conv2d(
                num_input_features,
                num_init_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.features.add_module("norm1", nn.BatchNorm2d(num_init_features))

    def forward(self, x):
        return self.features(x)


class _Concatenate(nn.Module):
    def __init__(self, skip_connections):
        super(_Concatenate, self).__init__()
        self.skip_connections = skip_connections

    def forward(self, x):
        return torch.cat([x, self.skip_connections.pop()], 1)


class _TransitionUp(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, skip_connections):
        super(_TransitionUp, self).__init__()

        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                num_output_features * 2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        self.add_module("upsample", nn.Upsample(scale_factor=2, mode="bilinear"))
        self.add_module("cat", _Concatenate(skip_connections))
        self.add_module("norm2", nn.BatchNorm2d(num_output_features * 4))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module(
            "conv2",
            nn.Conv2d(
                num_output_features * 4,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )


class DenseUNet(nn.Module):
    def __init__(
        self,
        n_classes,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        downsample=False,
        pretrained_encoder_uri=None,
        progress=None,
    ):
        super(DenseUNet, self).__init__()
        self.skip_connections = []
        self.encoder = _DenseUNetEncoder(
            self.skip_connections,
            growth_rate,
            block_config,
            num_init_features,
            bn_size,
            drop_rate,
            downsample,
        )
        self.decoder = _DenseUNetDecoder(
            self.skip_connections,
            growth_rate,
            block_config,
            num_init_features,
            bn_size,
            drop_rate,
            downsample,
        )
        self.classifier = nn.Conv2d(
            num_init_features, n_classes, kernel_size=1, stride=1, bias=True
        )
        self.softmax = nn.Softmax(dim=1)

        self.encoder._load_state_dict = self.encoder.load_state_dict
        self.encoder.load_state_dict = lambda state_dict: self.encoder._load_state_dict(
            state_dict, strict=False
        )
        if pretrained_encoder_uri:
            _load_state_dict(self.encoder, str(pretrained_encoder_uri), progress)
        self.encoder.load_state_dict = lambda state_dict: self.encoder._load_state_dict(
            state_dict, strict=True
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        y = self.classifier(x)
        return self.softmax(y)


"""
Code copied from https://github.com/Beckschen/TransUNet
"""
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(
        cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups
    )


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block."""

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(
            cmid, cmid, stride, bias=False
        )  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(
            weights[os.path.join(n_block, n_unit, "conv1/kernel")], conv=True
        )
        conv2_weight = np2th(
            weights[os.path.join(n_block, n_unit, "conv2/kernel")], conv=True
        )
        conv3_weight = np2th(
            weights[os.path.join(n_block, n_unit, "conv3/kernel")], conv=True
        )

        gn1_weight = np2th(weights[os.path.join(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[os.path.join(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[os.path.join(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[os.path.join(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[os.path.join(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[os.path.join(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, "downsample"):
            proj_conv_weight = np2th(
                weights[os.path.join(n_block, n_unit, "conv_proj/kernel")], conv=True
            )
            proj_gn_weight = np2th(
                weights[os.path.join(n_block, n_unit, "gn_proj/scale")]
            )
            proj_gn_bias = np2th(weights[os.path.join(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            3, width, kernel_size=7, stride=2, bias=False, padding=3
                        ),
                    ),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width, cout=width * 4, cmid=width
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 4, cout=width * 4, cmid=width
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 4,
                                            cout=width * 8,
                                            cmid=width * 2,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 8,
                                            cmid=width * 2,
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 16,
                                            cmid=width * 4,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 16,
                                            cout=width * 16,
                                            cmid=width * 4,
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(
                    x.size(), right_size
                )
                feat = torch.zeros(
                    (b, x.size()[1], right_size, right_size), device=x.device
                )
                feat[:, :, 0 : x.size()[2], 0 : x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = nn.modules.utils._pair(img_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (
                img_size[0] // 16 // grid_size[0],
                img_size[1] // 16 // grid_size[1],
            )
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (
                img_size[1] // patch_size_real[1]
            )
            self.hybrid = True
        else:
            patch_size = nn.modules.utils._pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )

        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = (
                np2th(weights[os.path.join(ROOT, ATTENTION_Q, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            key_weight = (
                np2th(weights[os.path.join(ROOT, ATTENTION_K, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            value_weight = (
                np2th(weights[os.path.join(ROOT, ATTENTION_V, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            out_weight = (
                np2th(weights[os.path.join(ROOT, ATTENTION_OUT, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )

            query_bias = np2th(weights[os.path.join(ROOT, ATTENTION_Q, "bias")]).view(
                -1
            )
            key_bias = np2th(weights[os.path.join(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[os.path.join(ROOT, ATTENTION_V, "bias")]).view(
                -1
            )
            out_bias = np2th(weights[os.path.join(ROOT, ATTENTION_OUT, "bias")]).view(
                -1
            )

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[os.path.join(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[os.path.join(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[os.path.join(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[os.path.join(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(
                np2th(weights[os.path.join(ROOT, ATTENTION_NORM, "scale")])
            )
            self.attention_norm.bias.copy_(
                np2th(weights[os.path.join(ROOT, ATTENTION_NORM, "bias")])
            )
            self.ffn_norm.weight.copy_(
                np2th(weights[os.path.join(ROOT, MLP_NORM, "scale")])
            )
            self.ffn_norm.bias.copy_(
                np2th(weights[os.path.join(ROOT, MLP_NORM, "bias")])
            )


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SODEDecoderBlock(nn.Module):
    def __init__(self, non_linearity, num_filters, tol, adjoint, counter) -> None:
        super().__init__()
        nf = num_filters
        self.counter = counter

        if self.counter == 0:
            self.initial_velocity = InitialVelocity(nf, non_linearity)

        ode_up1 = ConvSODEFunc(nf *3, False, non_linearity)
        self.odeblock_up1 = ODEBlock(ode_up1, tol=tol, adjoint=adjoint)
        self.conv_up_embed_1 = nn.Conv2d(nf *3 , (nf * 3)//2, 1, 1)
        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x, skip=None):
        if self.counter == 0:
            x = self.initial_velocity(x)

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        x = self.non_linearity(self.conv_up_embed_1(x))
        x = self.odeblock_up1(x)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config, sode_decoder=False):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(
                4 - self.config.n_skip
            ):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]
        if not sode_decoder:
            blocks = [
                DecoderBlock(in_ch, out_ch, sk_ch)
                for in_ch, out_ch, sk_ch in zip(
                    in_channels, out_channels, skip_channels
                )
            ]
        else:
            blocks = [
                SODEDecoderBlock(
                    non_linearity="lrelu",
                    num_filters=in_ch,
                    tol=1e-3,
                    adjoint=True,
                    counter=i
                )
                for i, in_ch in enumerate(in_channels)
            ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        (
            B,
            n_patch,
            hidden,
        ) = (
            hidden_states.size()
        )  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        img_size=224,
        num_classes=21843,
        zero_head=False,
        vis=False,
        sode_decoder=False,
    ):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config, sode_decoder=sode_decoder)
        self.segmentation_head = SegmentationHead(
            in_channels=config["decoder_channels"][-1],
            out_channels=config["n_classes"],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, _, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"])
            )

            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                # logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print("load_pretrained: grid-size from %s to %s" % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True)
                )
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for (
                    bname,
                    block,
                ) in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


if __name__ == "__main__":
    input_array_np = np.ones([16, 3, 256, 256])

    input_array = torch.from_numpy(input_array_np).float()
    print(1, input_array.size(), input_array.min(), input_array.max())
    
    model = Unet(depth=4, num_filters=3, output_dim=1, block="PLN").eval()
    output_array = model(input_array)
    print(1, output_array.size(), output_array.min(), output_array.max())


    input_array = torch.from_numpy(input_array_np).float()
    print(2, input_array.size(), input_array.min(), input_array.max())
    
    model = Unet(depth=4, num_filters=3, output_dim=1, block="RSE").eval()
    output_array = model(input_array)
    print(2, output_array.size(), output_array.min(), output_array.max())


    input_array = torch.from_numpy(input_array_np).float()
    print(3, input_array.size(), input_array.min(), input_array.max())
    
    model = Unet(depth=4, num_filters=3, output_dim=1, block="DSE").eval()
    output_array = model(input_array)
    print(3, output_array.size(), output_array.min(), output_array.max())


    input_array = torch.from_numpy(input_array_np).float()
    print(4, input_array.size(), input_array.min(), input_array.max())
    
    model = Unet(depth=4, num_filters=3, output_dim=1, block="INC").eval()
    output_array = model(input_array)
    print(4, output_array.size(), output_array.min(), output_array.max())


    input_array = torch.from_numpy(input_array_np).float()
    print(5, input_array.size(), input_array.min(), input_array.max())
    
    model = Unet(depth=4, num_filters=3, output_dim=1, block="PSP").eval()
    output_array = model(input_array)
    print(5, output_array.size(), output_array.min(), output_array.max())

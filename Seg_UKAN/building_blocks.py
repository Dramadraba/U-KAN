import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver


class InitialVelocity(nn.Module):
    def __init__(self, nf, non_linearity="relu"):
        super(InitialVelocity, self).__init__()

        self.norm1 = nn.InstanceNorm2d(nf)
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x0):
        out = self.norm1(x0)
        out = self.conv1(out)
        out = self.non_linearity(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        return torch.cat((x0, out), dim=1)

class InitialVelocity3D(nn.Module):
    def __init__(self, nf, non_linearity="relu"):
        super(InitialVelocity3D, self).__init__()

        self.norm1 = nn.InstanceNorm3d(nf)
        self.conv1 = nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm3d(nf)
        self.conv2 = nn.Conv3d(nf, nf, kernel_size=1, stride=1, padding=0)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x0):
        out = self.norm1(x0)
        out = self.conv1(out)
        out = self.non_linearity(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        return torch.cat((x0, out), dim=1)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-3, adjoint=False):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes
        Utility class that wraps odeint and odeint_adjoint.
        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver
            adjoint (bool): whether to use the adjoint method for gradient calculation
        """
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None, method="rk4"):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.adjoint:
            out = odeint_adjoint(
                self.odefunc,
                x,
                integration_time,
                rtol=self.tol,
                atol=self.tol,
                method=method,
                options={"max_num_steps": MAX_NUM_STEPS},
            )
        else:
            out = odeint(
                self.odefunc,
                x,
                integration_time,
                rtol=self.tol,
                atol=self.tol,
                method=method,
                options={"max_num_steps": MAX_NUM_STEPS},
            )

        if eval_times is None:
            return out[1]  # out[1][:int(len(x)/2)]  Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0.0, 1.0, timesteps)
        return self.forward(x, eval_times=integration_time)


class ConvODEFunc(nn.Module):
    def __init__(self, nf, time_dependent=False, non_linearity="relu"):
        """
        Block for ConvODEUNet
        Args:
            nf (int): number of filters for the conv layers
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvODEFunc, self).__init__()
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations

        if time_dependent:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
        else:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, t, x):
        self.nfe += 1
        if self.time_dependent:
            out = self.norm1(x)
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
        else:
            out = self.norm1(x)
            out = self.conv1(out)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
        return out


class Conv2dTime(nn.Conv2d):
    def __init__(self, in_channels, *args, **kwargs):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes
        Conv2d module where time gets concatenated as a feature map.
        Makes ODE func aware of the current time step.
        """
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        """
        Block for LevelBlock
        Args:
            in_channels (int): number of input filters for first conv layer
            out_channels (int): number of output filters for the last layer
        """
        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

class ConvBlock3D(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        """
        Block for LevelBlock
        Args:
            in_channels (int): number of input filters for first conv layer
            out_channels (int): number of output filters for the last layer
        """
        super().__init__(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=out_channels, 
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

class ConvSODEFunc(nn.Module):
    def __init__(self, nf, time_dependent=False, non_linearity="relu"):
        """
        Block for ConvSODEUNet
        Args:
            nf (int): number of filters for the conv layers
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvSODEFunc, self).__init__()
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations

        if time_dependent:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
        else:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            # changed to kernel_size 1 with padding 0 instead of 1
            self.conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, t, x):
        cutoff = int(x.shape[1] / 2)  # int(len(x)/2)
        z = x[:, :cutoff]
        v = x[:, cutoff:]
        into = torch.cat((z, v), dim=1)
        self.nfe += 1
        if self.time_dependent:
            out = self.norm1(into)
            out = self.conv1(t, into)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
        else:
            out = self.norm1(into)
            out = self.conv1(out)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
        return out

class ConvSODEFunc3D(nn.Module):
    def __init__(self, nf, time_dependent=False, non_linearity="relu"):
        """
        Block for ConvSODEUNet
        Args:
            nf (int): number of filters for the conv layers
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvSODEFunc3D, self).__init__()
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations


        self.norm1 = nn.InstanceNorm3d(nf)
        self.conv1 = nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm3d(nf)
        # changed to kernel_size 1 with padding 0 instead of 1
        self.conv2 = nn.Conv3d(nf, nf, kernel_size=1, stride=1, padding=0)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, t, x):
        cutoff = int(x.shape[1] / 2)  # int(len(x)/2)
        z = x[:, :cutoff]
        v = x[:, cutoff:]
        into = torch.cat((z, v), dim=1)
        self.nfe += 1
        if self.time_dependent:
            out = self.norm1(into)
            out = self.conv1(t, into)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
        else:
            out = self.norm1(into)
            out = self.conv1(out)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
        return out

class ConvResFunc(nn.Module):
    def __init__(self, num_filters, non_linearity="relu"):
        """
        Block for ConvResUNet
        Args:
            num_filters (int): number of filters for the conv layers
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvResFunc, self).__init__()

        self.conv1 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        self.norm = nn.InstanceNorm2d(2, num_filters)
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        out = self.norm(x)
        out = self.conv1(x)
        out = self.non_linearity(out)
        out = self.norm(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        out = x + out
        return out


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ConvDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_size=4, drop_rate=0, memory_efficient=False):
        super(ConvDenseBlock, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(in_channels)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * out_channels, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * out_channels)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        "Bottleneck function"
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):  # noqa: F811
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class INCConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(INCConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        out_1x1 = int(out_channels/4)
        red_3x3 = int(in_channels/2)
        out_3x3 = int(out_channels/4)
        red_5x5 = int(out_channels/8) 
        out_5x5 = int(out_channels/4)
        out_pool = int(out_channels/4)
        print(in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool)
        self.branch1 = INCConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            INCConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            INCConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            INCConvBlock(in_channels, red_5x5, kernel_size=1),
            INCConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            INCConvBlock(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)


class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in sizes])
        self.bottleneck = nn.Conv2d(in_channels * (len(sizes) + 1), out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, in_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class LevelBlock(nn.Module):
    def __init__(self, depth, total_depth, in_channels, out_channels, block):
        """
        Block for UNet
        Args:
            depth (int): current depth of blocks (starts with total_depth: n,...,0)
            total_depth (int): total_depth of U-Net
            in_channels (int): number of input filters for first conv layer
            out_channels (int): number of output filters for the last layer
            block (str): block type
        """
        super(LevelBlock, self).__init__()
        self.depth = depth
        self.total_depth = total_depth
        if depth > 1:
            if str(block) == "PLN":
                self.encode = ConvBlock(in_channels, out_channels)
            elif str(block) == "RSE":
                self.encode = ResidualBlock(in_channels, out_channels)
            elif str(block) == "DSE":
                self.encode = ConvDenseBlock(in_channels, out_channels)
            elif str(block) == "INC":
                self.encode = nn.Sequential(
                    INCConvBlock(in_channels, in_channels*4, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    InceptionBlock(in_channels*4, in_channels*4),
                    nn.ReLU(),
                    INCConvBlock(in_channels*4, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            elif str(block) == "PSP":
                self.encode = PSPBlock(in_channels, out_channels)
            else:
                raise Exception
            self.down = nn.MaxPool2d(2, 2)
            self.next = LevelBlock(
                depth - 1, total_depth, out_channels, out_channels * 2, block=block
            )
            next_out = list(self.next.modules())[-2].out_channels
            self.up = nn.ConvTranspose2d(next_out, next_out // 2, 2, 2)
            self.decode = ConvBlock(next_out // 2 + out_channels, out_channels)
        else:
            self.embed = ConvBlock(in_channels, out_channels)

    def forward(self, inp):
        if self.depth > 1:
            first_x = self.encode(inp)
            x = self.down(first_x)
            x = self.next(x)
            x = self.up(x)

            # center crop
            i_h = first_x.shape[2]
            i_w = first_x.shape[3]

            total_crop = i_h - x.shape[2]
            crop_left_top = total_crop // 2
            crop_right_bottom = total_crop - crop_left_top

            cropped_input = first_x[
                :,
                :,
                crop_left_top : i_h - crop_right_bottom,
                crop_left_top : i_w - crop_right_bottom,
            ]
            x = torch.cat((cropped_input, x), dim=1)

            x = self.decode(x)
        else:
            x = self.embed(inp)

        return x

class LevelBlock3D(nn.Module):
    def __init__(self, depth, total_depth, in_channels, out_channels, block):
        """
        Block for UNet
        Args:
            depth (int): current depth of blocks (starts with total_depth: n,...,0)
            total_depth (int): total_depth of U-Net
            in_channels (int): number of input filters for first conv layer
            out_channels (int): number of output filters for the last layer
            block (str): block type
        """
        super(LevelBlock3D, self).__init__()
        self.depth = depth
        self.total_depth = total_depth
        if depth > 1:
            if str(block) == "PLN":
                self.encode = ConvBlock3D(in_channels, out_channels)
            else:
                raise Exception
            self.down = nn.MaxPool3d(2, 2)
            self.next = LevelBlock3D(
                depth - 1, total_depth, out_channels, out_channels * 2, block=block
            )
            next_out = list(self.next.modules())[-2].out_channels
            self.up = nn.ConvTranspose3d(next_out, next_out // 2, 2, 2)
            self.decode = ConvBlock3D(next_out // 2 + out_channels, out_channels)
        else:
            self.embed = ConvBlock3D(in_channels, out_channels)

    def forward(self, inp):
        if self.depth > 1:
            first_x = self.encode(inp)
            x = self.down(first_x)
            x = self.next(x)
            x = self.up(x)

            # center crop
            i_h = first_x.shape[-2]
            i_w = first_x.shape[-2]

            total_crop = i_h - x.shape[-2]
            crop_left_top = total_crop // 2
            crop_right_bottom = total_crop - crop_left_top

            cropped_input = first_x[
                :,
                :,
                crop_left_top : i_h - crop_right_bottom,
                crop_left_top : i_w - crop_right_bottom,
            ]
            x = torch.cat((cropped_input, x), dim=1)

            x = self.decode(x)
        else:
            x = self.embed(inp)

        return x

def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "softplus":
        return nn.Softplus()
    elif name == "swish":
        return Swish(inplace=True)
    elif name == "lrelu":
        return nn.LeakyReLU()


class Swish(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)

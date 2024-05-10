import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
import math


class ERAM(nn.Module):
    def __init__(self, in_channels):
        super(ERAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        ## Channel Attention (CA) Mechanism
        mid_channels = int(in_channels // 2)
        self.linear1 = nn.Linear(in_channels, mid_channels)
        self.linear2 = nn.Linear(mid_channels, in_channels)

        ## Spatial Attention (SA) Mechanism
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_sa = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_i = self.conv1(x)
        x_i = self.relu(x_i)
        x_i = self.conv2(x_i)

        # Channel Attention
        global_avg = F.avg_pool2d(x_i, x_i.size()[2:])
        global_avg = global_avg.reshape(global_avg.size(0), global_avg.size(1))

        x_i_reshaped = x_i.reshape(x_i.size(0), x_i.size(1), -1)
        global_var = x_i_reshaped.var(2)
        S_ca = global_avg + global_var

        M_ca = self.linear1(S_ca)
        M_ca = self.relu(M_ca)
        M_ca = self.linear2(M_ca)

        # Spatial Attention
        M_sa = self.depthwise_conv(x_i)
        M_sa = self.relu(M_sa)
        M_sa = self.conv_sa(M_sa)

        # Combining Channel and Spatial Attentions
        M_ca = M_ca.unsqueeze(-1).unsqueeze(-1)
        combined = M_ca + M_sa
        combined = self.sigmoid(combined)
        
        # Getting enhancement feature values
        x_i_cap = combined * x_i
        
        out = x_i_cap + x
        return out
        

class Sub_Pixel_Convolution_Layer(nn.Module):
    def __init__(self, upscale_ratio, single_layer=True, in_channels=1):
        super(Sub_Pixel_Convolution_Layer, self).__init__()
        
        out_channels = in_channels * upscale_ratio * upscale_ratio
        self.relu = nn.ReLU()
        if single_layer == True:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_ratio)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pixel_shuffle(x)
        return x


class SRUN_SinglePass(nn.Module):

    def downsampling_block(self, num_filters):
        blocks = nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
                )
        return blocks

    def upsampling_block(self, in_filters, out_filters, num_erams):
        blocks = nn.ModuleList([])
        for i in range(num_erams):
            blocks.append(ERAM(in_filters))
        blocks.append(Sub_Pixel_Convolution_Layer(upscale_ratio=2, single_layer=True, in_channels=in_filters))
        blocks.append(nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=1))
        
        block = nn.Sequential(*blocks)
        return block
        
    def __init__(self, scale_factor, in_channels=1, filter_size=12, num_eram_layers=20):
        super(SRUN_SinglePass, self).__init__()
        
        num_blocks = int(math.log2(scale_factor))
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.upscale = Sub_Pixel_Convolution_Layer(upscale_ratio=scale_factor, single_layer=False, in_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, filter_size, kernel_size=3, stride=1, padding=1)

        for _ in range(num_blocks):
            self.down_blocks.append(self.downsampling_block(filter_size))

        for i in range(num_blocks):
            if i == 0:
                self.up_blocks.append(self.upsampling_block(filter_size, filter_size, num_eram_layers))
            else:
                self.up_blocks.append(self.upsampling_block(filter_size*2, filter_size, num_eram_layers))

        self.conv2 = nn.Conv2d(filter_size*2, 1, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.upscale(x)
        x = self.conv1(x)

        outs = [x]
        for i in range(len(self.down_blocks)):
            temp = outs[-1]
            result = self.down_blocks[i](temp)
            outs.append(result)

        upscale_outs = []
        result = self.up_blocks[0](outs[-1])
        upscale_outs.append(result)

        for i in range(1, len(self.up_blocks)):
            temp1 = upscale_outs[-1]
            temp2 = outs[-i-1]
            input = torch.cat([temp1, temp2], dim=1)
            result = self.up_blocks[i](input)
            upscale_outs.append(result)

        input = torch.cat([outs[0], upscale_outs[-1]], dim=1)
        output = self.conv2(input)
        output = self.activation(output)

        return output, outs + upscale_outs


if __name__ == '__main__':
    model = SRUN_SinglePass(scale_factor=8)
    model = model.cuda()
    inp = torch.randn(8,1,32,32).cuda()
    out,features = model(inp)
    # for i in features:
    #     print(i.size())
    print(out.size())
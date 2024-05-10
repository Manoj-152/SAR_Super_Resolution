import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_Block(nn.Module):
    def __init__(self, in_filters, mid_filters, kernel_size, stride, padding):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, mid_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(mid_filters)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(mid_filters, in_filters, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        out += identity
        return out


class SORTN(nn.Module):

    def downsampling_block(self, in_filters, out_filters):
        block = nn.Sequential(
                    nn.Conv2d(in_filters, out_filters, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(out_filters),
                    nn.LeakyReLU(0.2),
                )
        return block

    def upsampling_block(self, in_filters, out_filters, dropout):
        blocks = nn.ModuleList([
                        nn.ReLU(),
                        nn.ConvTranspose2d(in_filters, out_filters, kernel_size=7, stride=2, padding=3, output_padding=1),
                        nn.BatchNorm2d(out_filters),
                    ])

        if dropout == True:
            blocks.append(nn.Dropout(0.2))

        block = nn.Sequential(*blocks)
        return block

    def residual_block(self, in_filters, mid_filters):
        block = Residual_Block(in_filters, mid_filters, kernel_size=3, stride=1, padding=1)
        return block

    def __init__(self, num_scaling_blocks=4, num_residual_blocks=6, initial_filter_size=64):
        super(SORTN, self).__init__()
        
        self.blocks = nn.ModuleList([self.downsampling_block(1,initial_filter_size)])
        filter_size = initial_filter_size
        for i in range(1, num_scaling_blocks):
            self.blocks.append(self.downsampling_block(filter_size, filter_size*2))
            filter_size = filter_size * 2

        # Filter size is 8 times the initial filter size now
        for i in range(num_residual_blocks):
            self.blocks.append(self.residual_block(filter_size, filter_size))

        for i in range(1, num_scaling_blocks):
            if i == 1: dropout = False
            else: dropout = True
            self.blocks.append(self.upsampling_block(filter_size, int(filter_size/2), dropout))
            filter_size = int(filter_size/2)

        self.blocks.append(self.upsampling_block(filter_size, 3, dropout=False))
        # self.blocks.append(nn.Tanh())
        self.blocks.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.blocks)

    def forward(self, x):
        f_maps = []
        f_maps.append(x)
        for i,module in enumerate(self.blocks):
            x = module(x)
            f_maps.append(x)
        # x = self.model(x)
        return x, f_maps


if __name__ == '__main__':
    model = SORTN()
    model = model.cuda()
    inp = torch.randn(8,1,256,256).cuda()
    out,_ = model(inp)
    print(out.size())
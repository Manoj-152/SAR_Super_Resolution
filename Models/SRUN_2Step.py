
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class GLU(nn.Module):
    def __init__(self, res_enc_len, channels):
        super(GLU, self).__init__()
        self.linear = nn.Linear(res_enc_len, channels)

    def forward(self, x, res_enc):
        # nc = x.size(1)
        # assert nc % 2 == 0, "channels dont divide 2!"
        # nc = int(nc/2)
        enc = self.linear(res_enc)
        enc = enc.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return x * torch.sigmoid(enc*x)


class SRUN_2Step(nn.Module):

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
        super(SRUN_2Step, self).__init__()
        
        num_blocks = int(math.log2(scale_factor))
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.glus = nn.ModuleList([])
        self.upscale = Sub_Pixel_Convolution_Layer(upscale_ratio=scale_factor, single_layer=False, in_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, filter_size, kernel_size=3, stride=1, padding=1)
        self.input_encoder = Embedder(
            input_dims = 1,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.temp_enc = self.input_encoder(torch.Tensor([1]))
        self.res_enc_len = len(self.temp_enc)

        for _ in range(num_blocks):
            self.down_blocks.append(self.downsampling_block(filter_size))

        for i in range(num_blocks):
            if i == 0:
                self.up_blocks.append(self.upsampling_block(filter_size, filter_size, num_eram_layers))
            else:
                self.up_blocks.append(self.upsampling_block(filter_size*2, filter_size, num_eram_layers))
                self.glus.append(GLU(self.res_enc_len, filter_size))

        self.conv2 = nn.Conv2d(filter_size*2, 1, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x, res_label):
        self.input_enc = self.input_encoder(res_label).view(-1,self.res_enc_len)
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
            temp2 = self.glus[i-1](outs[-i-1], self.input_enc)
            input = torch.cat([temp1, temp2], dim=1)
            result = self.up_blocks[i](input)
            upscale_outs.append(result)

        input = torch.cat([outs[0], upscale_outs[-1]], dim=1)
        output = self.conv2(input)
        output = self.activation(output)

        return output, outs + upscale_outs


if __name__ == '__main__':
    model = SRUN_2Step(scale_factor=4)
    model = model.cuda()
    inp = torch.randn(8,1,64,64).cuda()
    res_label = torch.Tensor([1,2,4,1,2,4,1,2]).unsqueeze(1).cuda()
    out,features = model(inp, res_label)
    # for i in features:
    #     print(i.size())
    print(out.size())

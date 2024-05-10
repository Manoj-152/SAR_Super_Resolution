import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchGAN(nn.Module):
    def __init__(self, input_channels):
        super(PatchGAN, self).__init__()

        # Convolution layers below are such a way that the receptive field is brought to 70x70 pixels.
        # This means that each pixel in our final feature map represents a 70x70 patch of the input.
        self.model = nn.ModuleList([   nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ])

        self.model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        self.model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]
        
        self.model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        self.model += [  nn.Conv2d(512, 1, 4, padding=1) ]
        self.model += [  nn.Sigmoid()  ]

        #self.model = nn.Sequential(*self.model)

    def forward(self, x):
        f_maps = []
        f_maps.append(x)
        for i,layer in enumerate(self.model):
            x = layer(x)
            f_maps.append(x)
        #x = self.model(x)
        # print(x.size())
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1), f_maps

if __name__ == '__main__':
    model = PatchGAN(3)
    inp = torch.randn([8,3,256,256])
    out,_ = model(inp)
    print(out.size())

import torch
import torch.nn as nn
import torchvision


class NonLocalBlock(nn.Module):
    def __init__(self, channel1,channel2):
        super(NonLocalBlock, self).__init__()
        self.inter_channel1 = channel1 // 2
        self.inter_channel2 = channel2 // 2
        self.conv_phi = nn.Conv2d(in_channels=channel2, out_channels=self.inter_channel2, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel1, out_channels=self.inter_channel1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel1, out_channels=self.inter_channel1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel2, out_channels=channel2, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        # self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=self.inter_channel // 2, kernel_size=1,
        #                            stride=1, padding=0, bias=False)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, xl,xh):
        # [N, C, H , W]
        b, c, h, w = xl.size()
        # print(xl.shape)
        # print(xh.shape)
        # [N, C/2, H * W]
        x_phi = self.conv_phi(xh).view(b, c, -1)
        # print(x_phi.shape)
        # [N, H * W, C/2]
        # x_theta = self.conv_theta(xl).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_theta = self.conv_theta(xl).view(b, c, -1)

        # x_theta = torch.transpose(x_theta, 1, 2).contiguous()
        # x_g = self.conv_g(xl).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(xl).view(b, c, -1)

        # x_g = torch.transpose(x_g, 1, 2).contiguous()

        # [N, H * W, H * W]
        mul_theta_phi = torch.mul(x_theta, x_phi)

        mul_theta_phi = self.softmax(mul_theta_phi)

        # [N, H * W, C/2]
        mul_theta_phi_g = torch.mul(mul_theta_phi, x_g)
        # print(mul_theta_phi_g.shape)
        mul_theta_phi_g = mul_theta_phi_g.contiguous().view(b,self.inter_channel2, h, w)
        # [N, C/2, H, W]
        mask = self.conv_mask(mul_theta_phi_g)
        # print(mask.shape)

        out = mask + xl
        return out


# if __name__=='__main__':
#     model = NonLocalBlock(channel=16)
#     print(model)
#
#     input1 = torch.randn(1, 16, 64, 64)
#     input2= torch.randn(1, 16, 64, 64)
#     out = model(input1,input2)
#     print(out.shape)

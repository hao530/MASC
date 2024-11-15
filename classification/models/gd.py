import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
class FeatureFusion(nn.Module):
    def __init__(self, low_feat_size, high_feat_size):
        super(FeatureFusion, self).__init__()

        # Assuming low_feat_size and high_feat_size are the dimensions of low-level and high-level features
        self.gate_param = nn.Parameter(torch.zeros(1))  # Initialize the gate parameter b to 0
        self.dropout = nn.Dropout(p=0.5)  # Dropout with a probability of 0.5
        self.relu = nn.ReLU(False)

    def forward(self, low_feat, high_feat):
        # Resize high-level features to match the size of low-level features
        #high_feat_resized = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=False)

        # Generate gate mapping G
        gate_map = self.dropout(torch.ones_like(low_feat) * self.gate_param)
        gate_map = gate_map.clone()
        # Dynamic fusion of high-level and low-level features
        fused_features = gate_map * low_feat + (1 - gate_map) * high_feat

        activated_features = self.relu(fused_features)

        return activated_features
#if __name__=='__main__':
#    model = FeatureFusion(low_feat_size=(1, 16, 64, 64), high_feat_size=(1, 16, 64, 64))
#    print(model)

#    input1 = torch.randn(1, 16, 64, 64)
#    input2= torch.randn(1, 16, 64, 64)
#    out = model(input1,input2)
#    print(out)
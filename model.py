# model.py
# 作用：定义YOLO-like检测网络结构（轻量Backbone + Head）
# 输出张量形状：(B, S, S, A, 5+C)，便于后续手写decode/loss

import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class YoloTiny(nn.Module):
    """
    简化YOLO-like:
    - backbone把640降到S=20（5次stride=2）
    - head输出A*(5+C)
    """
    def __init__(self, num_classes=3, S=20, A=3):
        super().__init__()
        self.num_classes = num_classes
        self.S = S
        self.A = A
        out_ch = A * (5 + num_classes)

        self.backbone = nn.Sequential(
            ConvBNAct(3, 16, 3, 2, 1),    # 320
            ConvBNAct(16, 32, 3, 2, 1),   # 160
            ConvBNAct(32, 64, 3, 2, 1),   # 80
            ConvBNAct(64, 128, 3, 2, 1),  # 40
            ConvBNAct(128, 256, 3, 2, 1), # 20 -> S
        )
        self.head = nn.Conv2d(256, out_ch, 1, 1, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)  # (B, A*(5+C), S, S)
        B, _, S, S2 = x.shape
        assert S == self.S and S2 == self.S
        x = x.permute(0, 2, 3, 1).contiguous()          # (B,S,S, A*(5+C))
        x = x.view(B, S, S, self.A, 5 + self.num_classes)
        return x

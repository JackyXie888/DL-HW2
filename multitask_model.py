
import torch
import torch.nn as nn
from ultralytics import YOLO

import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOv8Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 直接載入預訓練權重（預設載到 CPU 或自動選 GPU）
        base = YOLO("yolov8n.pt")           # 取出 Ultralytics YOLO 包裝器
        layers = list(base.model.model.children())

        # 分成 stem / low / mid / high 四個階段
        self.stem = nn.Sequential(*layers[0:4])
        self.low  = nn.Sequential(*layers[4:6])
        self.mid  = nn.Sequential(*layers[6:8])
        self.high = nn.Sequential(*layers[8:10])

    def forward(self, x):
        # x: (B, 3, 224, 224)，輸入可在外層自行處理 device
        x   = self.stem(x)    # → (B,  64, 112, 112)
        low = self.low(x)     # → (B, 128,  56,  56)
        mid = self.mid(low)   # → (B, 256,  28,  28)
        high= self.high(mid)  # → (B, 256,   7,   7)
        return low, mid, high

class ConvNeck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.neck = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, in_ch, 7, 7)
        return self.neck(x)   # → (B, out_ch, 7, 7)

class UnifiedHead(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_cls: int,
                 num_seg: int,
                 num_det_cls: int,
                 num_anchors: int):
        super().__init__()
        # 1. 分類 Head (Global Average Pool → Linear)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # (B, in_ch, 1, 1)
            nn.Flatten(),                # (B, in_ch)
            nn.Linear(in_ch, num_cls)    # (B, num_cls)
        )
        # 2. 分割 Head (conv → conv → upsample)
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_ch, num_seg, kernel_size=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        )
        # 3. 偵測 Head (conv → conv)
        # 輸出維度：A * (4 + 1 + num_det_cls)，A 是錨點數
        out_channels = num_anchors * (4 + 1 + num_det_cls)
        self.det_head = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, in_ch, 7, 7)
        cls_out = self.cls_head(x)   # → (B, num_cls)
        seg_out = self.seg_head(x)   # → (B, num_seg, 28, 28)
        det_out = self.det_head(x)   # → (B, A*(4+1+num_det_cls), 7, 7)
        return cls_out, seg_out, det_out

class MultiTaskUnifiedModel(nn.Module):
    def __init__(self,
                 num_cls: int = 10,
                 num_seg: int = 21,
                 num_det_cls: int = 10,
                 num_anchors: int = 3):
        super().__init__()
        # Backbone → 輸出 third-stage 特徵 (high)
        self.backbone = YOLOv8Backbone()
        # Neck: 把 backbone 最後輸出通道套成 256 → 256
        self.neck     = ConvNeck(in_ch=256, out_ch=256)
        # Unified Head：分類 / 分割 / 偵測
        self.head     = UnifiedHead(
            in_ch=256,
            num_cls=num_cls,
            num_seg=num_seg,
            num_det_cls=num_det_cls,
            num_anchors=num_anchors
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)，假定呼叫者自行把 x.to(device)
        _, _, high = self.backbone(x)      # high: (B, 256, 7, 7)
        neck_out   = self.neck(high)       # (B, 256, 7, 7)
        return self.head(neck_out)         # 回傳 (cls_out, seg_out, det_out)



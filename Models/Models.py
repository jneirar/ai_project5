import torch
import torch.nn as nn

class Lenet5_4_fc(nn.Module):
    def __init__(self):
        super(Lenet5_4_fc, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.AdaptiveAvgPool2d(output_size=(5, 5))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace = False),
            nn.Linear(in_features=400, out_features=200, bias=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.2, inplace = False),
            nn.Linear(in_features=200, out_features=100, bias=True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features=100, out_features=50, bias=True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features=50, out_features=26, bias=True)
         )
        
    def forward(self, image):
        out = self.features(image)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    
class Lenet5_1_fc(nn.Module):
    def __init__(self):
        super(Lenet5_1_fc, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.AdaptiveAvgPool2d(output_size=(5, 5))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace = False),
            nn.Linear(in_features=400, out_features=26, bias=True)
         )
        
    def forward(self, image):
        out = self.features(image)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    
class AlexNet_5_fc(nn.Module):
    def __init__(self):
        super(AlexNet_5_fc, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
            nn.AdaptiveAvgPool2d(output_size=(6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace = False),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.2, inplace = False),
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features=1024, out_features=256, bias=True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features=256, out_features=26, bias=True)
         )

    def forward(self, image):
        out = self.features(image)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    
class AlexNet_2_fc(nn.Module):
    def __init__(self):
        super(AlexNet_2_fc, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
            nn.AdaptiveAvgPool2d(output_size=(6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace = False),
            nn.Linear(in_features=9216, out_features=2048, bias=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.2, inplace = False),
            nn.Linear(in_features=2048, out_features=26, bias=True)
         )

    def forward(self, image):
        out = self.features(image)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

import torch.nn as nn
import torchvision.models as models


class LandmarkModel(nn.Module):
    def __init__(self, pts):
        super(LandmarkModel, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2 * pts)
        self.fc_1 = nn.Linear(2 * pts, 50 * pts)
        self.fc_2 = nn.Linear(50 * pts, 2 * pts)
        
    def forward(self, x):
        x = self.model(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        
        return x
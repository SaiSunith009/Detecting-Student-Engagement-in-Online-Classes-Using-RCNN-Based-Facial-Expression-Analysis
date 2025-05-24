import torch
import torch.nn as nn
from torchvision import models

class RCNN(nn.Module):
    def __init__(self, num_classes=4, rnn_hidden_size=256, num_rnn_layers=2, dropout=0.5):
        super(RCNN, self).__init__()
        
        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn.classifier = nn.Identity()
        
        self.feature_size = 1280  # MobileNetV2 feature size
        
        self.rnn = nn.GRU(
            input_size=self.feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.cnn(x)
        
        x = x.view(batch_size, seq_length, -1)
        x, _ = self.rnn(x)
        
        x = x[:, -1, :]
        x = self.classifier(x)
        
        return x

def load_model(model_path):
    model = RCNN()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

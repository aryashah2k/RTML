# For our puffer surver we need to browse via a proxy!!
import os
# Set HTTP and HTTPS proxy
os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    '''
    An AlexNet-like CNN with Local Response Normalization (LRN)

    Attributes
    ----------
    num_classes : int
        Number of classes in the final multinomial output layer
    features : Sequential
        The feature extraction portion of the network
    avgpool : AdaptiveAvgPool2d
        Convert the final feature layer to 6x6 feature maps by average pooling if they are not already 6x6
    classifier : Sequential
        Classify the feature maps into num_classes classes
    use_lrn : bool
        Whether to use Local Response Normalization
    '''
    def __init__(self, num_classes: int = 10, use_lrn: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_lrn = use_lrn
        
        # First conv layer
        self.features_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
        )
        
        # First LRN layer (after ReLU, before MaxPool)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        
        # First MaxPool and second conv
        self.features_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        
        # Second LRN layer (after ReLU, before MaxPool)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        
        # Second MaxPool and remaining layers
        self.features_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Added dropout rate as per paper
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Added dropout rate as per paper
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv + ReLU
        x = self.features_1(x)
        
        # First LRN (if enabled)
        if self.use_lrn:
            x = self.lrn1(x)
            
        # First MaxPool + second conv + ReLU
        x = self.features_2(x)
        
        # Second LRN (if enabled)
        if self.use_lrn:
            x = self.lrn2(x)
            
        # Remaining layers
        x = self.features_3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

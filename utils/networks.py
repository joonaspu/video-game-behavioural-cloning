import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Mnih2015(nn.Module):
    """CNN head similar to one used in Mnih 2015
       (Human-level control through deep reinforcement learning, Mnih 2015)"""
    def __init__(self, image_shape, num_channels, num_actions):
        super(Mnih2015, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(num_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        c_out = self.conv3(self.conv2(self.conv1(torch.randn(1, num_channels, *image_shape))))
        self.conv3_size = np.prod(c_out.shape)
        print("conv3: {}".format(self.conv3_size))

        self.fc1 = nn.Linear(self.conv3_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.conv3_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
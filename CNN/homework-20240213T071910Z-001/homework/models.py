import torch
import torch.nn as nn
import IPython


class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[], n_input_channels=3, kernel_size=3):
        super().__init__()

        L = []
        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size=7, padding=1))
            L.append(torch.nn.ReLU())
            torch.nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
            L.append(torch.nn.ReLU())
            c = l
        L.append(torch.nn.Conv2d(c, 6, kernel_size=3))
        self.layers = torch.nn.Sequential(*L)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        output = self.layers(x).mean([2, 3])
        return output


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    #state_dict = torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'),map_location='cpu')
    #r.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'),map_location='cpu'))
    return r

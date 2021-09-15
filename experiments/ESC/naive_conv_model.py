import torch
import torch.nn as nn

class NaiveModel(nn.Module):

    def __init__(self, n_classes, n_filters=32, n_hidden=32):
        super(type(self), self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(1, n_filters, (1, 5), (1, 3))
        self.conv2 = nn.Conv2d(n_filters, n_filters, (1, 5), (1, 3))
        self.global_time_pooling = nn.AdaptiveMaxPool2d((None, 1))
        self.linear_size = n_filters*64
        self.linear1 = nn.Linear(self.linear_size, n_hidden)    
        self.linear2 = nn.Linear(n_hidden, n_classes) 

    def forward(self, x):
        x = x['mel_transform']
        z = self.activation(self.conv1(x))
        z = self.activation(self.conv2(z))
        z = self.global_time_pooling(z)
        z = self.activation(self.linear1(z.view(-1, self.linear_size)))
        return self.linear2(z)

    def create_trace(self, path='traced_model.pt'):
        dummy_example = {'mel_transform': torch.randn(10, 1, 64, 500)}
        traced_model = torch.jit.trace(self, (dummy_example))
        traced_model.save(path)


    
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        input = input.unsqueeze(0)
        rr,hn = self.rnn(input,hidden)
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers,1,self.hidden_size)

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,input,hidden,c):
        input = input.unsqueeze(0)
        rr,hn,cn = self.lstm(input,(hidden,c))
        return self.softmax(self.linear(rr)), hn, cn

    def initHidden(self):
        cn=hidden = torch.zeros(self.num_layers,1,self.hidden_size)
        return hidden,cn

class GRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super(GRU,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        def forward(self,input,hidden):
            input = input.unsqueeze(0)
            rr,hn = self.gru(input,hidden)
            return self.softmax(self.linear(rr)), hn

        def initHidden(self):
            return torch.zeros(self.num_layers,1,self.hidden_size)









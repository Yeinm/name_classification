import model
from train import rnn, lstm, gru


#评估模型


def evaluateRNN(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output.squeeze(0)

def evaluateLSTM(line_tensor):
    hidden,c= lstm.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i],hidden,c)
    return output.squeeze(0)

def evaluateGRU(line_tensor):
    hidden = gru.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i],hidden)
    return output.squeeze(0)


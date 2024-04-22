import model
from pre_deal import line_to_tensor
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

line='Bai'
line_tensor=line_to_tensor(line)
rnn_output=evaluateRNN(line_tensor)
lstm_output=evaluateLSTM(line_tensor)
gru_output=evaluateGRU(line_tensor)
print("rnn_output",rnn_output)
print("lstm_output",lstm_output)
print("gru_output",gru_output)




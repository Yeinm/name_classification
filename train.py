#实例化参数
import math
import random
import time

import torch
from matplotlib import pyplot as plt

from torch import nn

import pre_deal
import model

input_size= pre_deal.n_letters
n_hidden = 128
output_size= pre_deal.n_categories
num_layers = 1

input= pre_deal.line_to_tensor('B').squeeze(0)
hidden= c= torch.zeros(1,1,n_hidden)

rnn = model.RNN(input_size,n_hidden,output_size,num_layers)
lstm = model.LSTM(input_size,n_hidden,output_size)
gru= model.GRU(input_size,n_hidden,output_size)


rnn_outputs, next_hidden = rnn(input,hidden)
#print('rnn:',rnn_outputs)

lstm_outputs, next_hidden, c = lstm(input,hidden,c)
#print('lstm:',lstm_outputs)

gru_outputs, next_hidden = gru(input,hidden)
#print('gru:',gru_outputs)

#取最可能的类别的索引和其名称
def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return pre_deal.all_categories[category_i], category_i



#随机生成训练数据
def randomTrainingExample():
    category = random.choice(pre_deal.all_categories)
    line = random.choice(pre_deal.category_lines[category])
    category_tensor = torch.tensor([pre_deal.all_categories.index(category)], dtype=torch.long)
    line_tensor = pre_deal.line_to_tensor(line)
    return category, line, category_tensor, line_tensor
'''
for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('/category:',category, '/line:',line, '/category:',category_tensor, '/line_tensor:',line_tensor)
    print(line_tensor.shape)
'''
#初始化在这
criterion = nn.NLLLoss()
learning_rate = 0.01

def trainrnn(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()

def trainLSTM(category_tensor, line_tensor):
    hidden,c= lstm.initHidden()
    lstm.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i],hidden,c)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()
    for p in lstm.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()

def trainGRU(category_tensor, line_tensor):
    hidden = gru.initHidden()
    gru.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i],hidden)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()
    for p in gru.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()


#辅助函数时间计算
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#这里可以开始训练了
n_iters = 1000
#输出间隔
print_every=50
#绘图间隔
plot_every=10

def train(train_func):
#损失保存列表
    all_losses = []
    start = time.time()
    current_loss = 0
#迭代
    for iteration in range(1, n_iters+1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train_func(category_tensor, line_tensor)
        current_loss += loss

        if iteration % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iteration, iteration / n_iters * 100, timeSince(start), loss, line, guess, correct))

        if iteration % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    return all_losses,int(time.time()-start)

all_losses1, time1 = train(trainrnn)
all_losses2, time2 = train(trainLSTM)
all_losses3, time3 = train(trainGRU)

'''
#画布0
plt.figure(0)
plt.plot(all_losses1,label='RNN')
plt.plot(all_losses2,color='red',label='LSTM')
plt.plot(all_losses3,color='orange',label='GRU')
plt.legend(loc='upper right')

#画布1
plt.figure(1)
x_data=['RNN','LSTM','GRU']
y_data=[time1,time2,time3]
plt.bar(range(len(x_data)),y_data,tick_label=x_data)
'''



















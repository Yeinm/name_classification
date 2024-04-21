#实例化参数
import random
import torch
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

for i in range(1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('/category:',category, '/line:',line, '/category:',category_tensor, '/line_tensor:',line_tensor)
    print(line_tensor.shape)














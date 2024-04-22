import torch

import model
import evaluate
import pre_deal
from pre_deal import line_to_tensor


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate_fn(line_to_tensor(input_line))
        topv, topi = output.topk(3, 1, True)
        predictions = []
        for i in range(3):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, pre_deal.all_categories[category_index]))
            predictions.append([value, pre_deal.all_categories[category_index]])

 #调用
for evaluate_fn in [evaluate.evaluateRNN, evaluate.evaluateLSTM, evaluate.evaluateGRU]:
    print('-'*18)
    predict('Dovesky',evaluate_fn)
    predict('Jackson',evaluate_fn)
    predict('Satoshi',evaluate_fn)




























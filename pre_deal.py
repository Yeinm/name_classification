import glob
import os
import string

import torch
import unicodedata


data_path='./data/names/'

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# 将unicode编码转化为ascii编码
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# 读取文件
def readLines(filename):
    lines = open(filename,encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

# 读取数据
category_lines={}
all_categories=[]


# 读取文件
for filename in glob.glob(data_path+'*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories=len(all_categories)

#将人名转化为one-hot编码
def line_to_tensor(line):
    tensor = torch.zeros(len(line),1,n_letters)
    for li,letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


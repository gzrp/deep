import re
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torch.utils.data import DataLoader


# 网络结构，特征大小 embed_dim = 20 的词向量的嵌入层开始，加入一个 LSTM 的循环层，再加入两个全连接层，一个作为隐藏层，一个作为输出层，然后经过 sigmod 激活，输出层返回类别概率的预测值
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        out = self.embedding(x)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

train_dataset = IMDB(split='train')
test_dataset = IMDB(split='test')
torch.manual_seed(1)
train_dataset, valid_dataset = torch.utils.data.random_split(list(train_dataset), [20000, 5000])

def tokenizer(text):
    # 文本 中包含 html 标记、标点符号，非字母字符等
    text = re.sub('<[^>]*>', '', text)  # 将 text 中复合模型 pattern 的替换成空串，杀出 html 中的 <p> </p> 这类的标签
    # 查找并记录表情符号非字母字符
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)', text.lower())
    text = re.sub('[\\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')  # 删除非单词字符 匹配任意不是字母，数字，下划线，汉字的字符
    # 最终将表情符号拼接在文档末尾，为了保持一直，删除了表示鼻子的符号 -
    tokenized = text.split()
    return tokenized


token_counts = Counter()
for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)

# step3： 将每个 word 映射成一个整数, 及有 index:world          按照频率进行排序，编码
sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab = vocab(ordered_dict)
# 验证数据集和测试数据集中可能存在一些没有在训练数据集中出现的词，将不再 token_counts 中的词元对应的整数设置为 1 unk, 填充词元为 0 用于调整序列长度
vocab.insert_token("<pad>", 0) # 填充词元
vocab.insert_token("<unk>", 1) # 未知词元
vocab.set_default_index(1)
# 转化所有文本

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
lable_pipeline = lambda x: 0. if x == 1 else 1.

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        # _lable 1 neg / 2 pos
        label_list.append(lable_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))

    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True) # 填充，使得序列长度一致
    return padded_text_list, label_list, lengths

batch_size = 32
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64

model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

def train(dataloader):
    model.train()
    total_acc, total_loss = 0, 0
    total_size = 0
    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        #  [32 *1]  ==> [32]
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)
        total_size += label_batch.size(0)
        if total_size % 320 == 0:
            print(f'total_size={total_size} total_acc={total_acc}, total_loss={total_loss}')

    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)


def evalute(dataloader):
    model.eval()
    i = 0
    total_acc, total_loss = 0, 0
    total_size = 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)
            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)
            total_size += label_batch.size(0)
            if total_size % 320 == 0:
                print(f'total_size={total_size} total_acc={total_acc}, total_loss={total_loss}')

    return total_acc/total_size, total_loss/total_size

num_epochs = 10

for epoch in range(num_epochs):
    acc_train, loss_train = train(train_dl)
    print(f'Epoch {epoch + 1}/{num_epochs}: accuracy: {acc_train:.4f} loss: {loss_train:.4f}')
    acc_val, loss_val = evalute(valid_dl)
    print(f'Epoch {epoch + 1}/{num_epochs}: accuracy: {acc_val:.4f} loss: {loss_val:.4f}')

acc_test, loss_test = evalute(test_dl)
print(f'test accuracy:  {acc_test:.4f}')

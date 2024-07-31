import numpy as np
import random
from singa import device
from singa import tensor
from singa import opt
from model import Transformer

print("step0: 开始准备数据...")
# 数据集生成
soundmark = ['ei', 'bi:', 'si:', 'di:', 'i:', 'ef', 'dʒi:', 'eit∫', 'ai', 'dʒei', 'kei', 'el', 'em', 'en', 'əu', 'pi:',
             'kju:', 'ɑ:', 'es', 'ti:', 'ju:', 'vi:', 'd^blju:', 'eks', 'wai', 'zi:']

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


t = 100 #总条数
r = 0.9   #扰动项
seq_len = 6
src_tokens, tgt_tokens = [],[] #原始序列、目标序列列表


for i in range(t):
    src, tgt = [], []
    for j in range(seq_len):
        ind = random.randint(0, 25)
        src.append(soundmark[ind])
        if random.random() < r:
            tgt.append(alphabet[ind])
        else:
            tgt.append(alphabet[random.randint(0, 25)])
    src_tokens.append(src)
    tgt_tokens.append(tgt)


# 构建词表
from collections import Counter
flatten = lambda l: [item for sublist in l for item in sublist] # 展平二维数组


class Vocab:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        # 词元按照词频排序
        self.token2index.update({
            token: index + 4
            for index, (token, freq) in enumerate(
                sorted(Counter(flatten(self.tokens)).items(), key=lambda x: x[1] , reverse=True))
        })
        self.index2token = {index: token for token, index in self.token2index.items()}

    def __getitem__(self, query):
        if isinstance(query, (str, int)):
            if isinstance(query, str):
                return self.token2index.get(query, 3)
            elif isinstance(query, (int)):
                return self.index2token.get(query, '<unk>')
        elif isinstance(query, (list, tuple)):
            return [self.__getitem__(item) for item in query]

    def __len__(self):
        return len(self.index2token)

# 数据集构造
src_vocab , tgt_vocab = Vocab(src_tokens), Vocab(tgt_tokens)
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# 增加标识
encoder_input_np = np.asarray([src_vocab[line+['<pad>']] for line in src_tokens], dtype=np.int32)
decoder_input_np = np.asarray([tgt_vocab[['<bos>']+line] for line in tgt_tokens], dtype=np.int32)
decoder_output_np = np.asarray([tgt_vocab[line+['<eos>']] for line in tgt_tokens], dtype=np.int32)

# 记录损失变化
loss_history = []
# 训练 10 轮
num_epochs = 10
# 每次小批量 16
batch_size = 4
# 训练数据集合测试数据集 8:2
train_size = int(len(encoder_input_np) * 0.8)
test_size = len(encoder_input_np) - train_size

# 划分数据集
train_encoder_input = encoder_input_np[:train_size]
train_decoder_input = decoder_input_np[:train_size]
train_decoder_output = decoder_output_np[:train_size]

test_encoder_input = encoder_input_np[-test_size:]
test_decoder_input = decoder_input_np[-test_size:]
test_decoder_output = decoder_output_np[-test_size:]

print("step1: 数据准备完毕...")


# [batch_size, src_len]
src_len = train_encoder_input.shape[1]  # 7
tgt_len = train_decoder_output.shape[1]
# 此处 check failure stack trace   self.data = CTensor(list(shape), device, dtype) 存在问题


num_train_batch = train_encoder_input.shape[0] // batch_size
num_test_batch = test_encoder_input.shape[0] // batch_size
idx = np.arange(train_encoder_input.shape[0], dtype=np.int32)


def train():
    # 配置设备
    dev = device.create_cpu_device()
    # 设置随机种子
    dev.SetRandSeed(0)
    np.random.seed(0)

    # 模型
    model = Transformer(src_n_token=src_vocab_size,
                        tgt_n_token=tgt_vocab_size,
                        d_model=512,
                        n_head=8,
                        dim_feedforward=2048,
                        n_layers=6)

    # 优化器
    optimizer = opt.SGD(lr=0.001, momentum=0.9, weight_decay=1e-5)
    model.set_optimizer(optimizer)
    print("step3: 模型加载完毕...")
    # model.compile([tx_enc_inputs, tx_dec_inputs], is_train=True)

    tx_enc_inputs = tensor.Tensor((batch_size, src_len), dev, tensor.int32)
    tx_dec_inputs = tensor.Tensor((batch_size, tgt_len), dev, tensor.int32)
    ty_dec_outputs = tensor.Tensor((batch_size, tgt_len), dev, tensor.int32)
    # 训练与验证循环
    for epoch in range(num_epochs):
        # 训练正确率，测试正确率，训练损失
        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        # 训练模式
        model.train()
        model.graph(mode=False, sequential=False)
        for b in range(num_train_batch):
            print("epoch: ", epoch, "batch: ", b)
            # 获取一批数据
            x_enc_inputs = train_encoder_input[idx[b*batch_size:(b+1)*batch_size]]
            x_dec_inputs = train_decoder_input[idx[b*batch_size:(b+1)*batch_size]]
            y_dec_outputs = train_decoder_output[idx[b*batch_size:(b+1)*batch_size]]

            # 拷贝批数据到 input tensors
            tx_enc_inputs.copy_from_numpy(x_enc_inputs)
            tx_dec_inputs.copy_from_numpy(x_dec_inputs)
            ty_dec_outputs.copy_from_numpy(y_dec_outputs)

            out, loss = model(tx_enc_inputs, tx_dec_inputs, ty_dec_outputs)

            # 计算训练模式下的正确数量和损失

            train_loss += tensor.to_numpy(loss)[0]


if __name__ == '__main__':
    train()

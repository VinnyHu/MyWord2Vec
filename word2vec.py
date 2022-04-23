import numpy as np
import math
import random
class Word2Vec:

    def __init__(self, config, vocab_word, vocab_index, hs = 0, neg = 1, lr = 0.025):

        self.hs = hs
        self.neg = neg
        self.config = config
        self.vocab_word = vocab_word
        self.vocab_index = vocab_index
        self.lr = lr
        self.negative = config.negative
        self.window = config.window
        # init net parameters
        self.word_emb = np.random.uniform(low=-0.5, high=0.5, size=(self.config.vocab_size, self.config.emb_size))
        self.output_emb = np.zeros((self.config.vocab_size, self.config.emb_size),dtype=float)

        self.expTable = self.initExpTable(EXP_TABLE_SIZE=config.EXP_TABLE_SIZE, MAX_EXP=config.MAX_EXP)
        self.negTable = self.initUnigramTable(table_size=config.table_size,vocab_size=len(vocab_word))

        total_words_cn = 0
        for _, word_count in vocab_word:
            total_words_cn += word_count

        self.train_file = []
        with open(config.train_file) as f:
            for item in f.readlines():
                words = []
                for word in item.strip().split(' '):
                    if word not in vocab_index:
                        continue
                    cur_word_cn = vocab_word[vocab_index[word]][-1]
                    # 处理高频词
                    if config.sample > 0:
                        ran = math.sqrt(config.sample * total_words_cn / cur_word_cn) + config.sample * total_words_cn / cur_word_cn
                        random_num = random.uniform()
                        if ran < random_num:
                            continue
                    words.append(word)
                    if len(words) == config.MAX_SENTENCE_LENGTH:
                        break
                self.train_file.append(words)



    def train_CBOW(self):

        for sentence in self.train_file:
            sentence_length = len(sentence)
            for sentence_position, word in enumerate(sentence):
                nue1 = np.zeros(shape=(self.config.emb_size),dtype=float)
                nue1e = np.zeros(shape=(self.config.emb_size),dtype=float)

                b = random.randint(0, self.window - 1)
                cw = 0

                for a in range(b, self.window * 2 + 1 - b):
                    if a != self.window:
                        c = sentence_position - self.window + a
                        if c < 0:
                            continue
                        if c >= sentence_length:
                            continue
                        context_word = self.vocab_index[sentence[c]]
                        nue1 += self.word_emb[context_word]

                        # 记录有效上下文词个数
                        cw += 1
                word_vocab_index = self.vocab_index[word]
                word_unit = self.vocab_word[word_vocab_index][0]
                if cw > 0:
                    nue1 = nue1 / cw
                    if self.neg:
                        target = -1
                        label = -1
                        for d in range(self.negative + 1):
                            f = 0
                            if d == 0:
                                target = word_vocab_index
                                label = 1
                            else:
                                r = random.randint(1, self.config.table_size)
                                target = self.negTable[r]
                                if target == word_vocab_index:
                                    continue
                                label = 0

                            non_leaf_node_index = word_unit
                            f += np.matmul(nue1, self.output_emb[non_leaf_node_index])

                            if f <= self.config.MAX_EXP:
                                g = (label - 0) * self.lr
                            elif f >= self.config.MAX_EXP:
                                g = (label - 1) * self.lr
                            else:
                                g = (label - self.expTable[int((f + self.config.MAX_EXP) * (self.config.EXP_TABLE_SIZE / self.config.MAX_EXP / 2))]) * self.lr

                            # 计算累计误差
                            nue1e += g * self.output_emb[non_leaf_node_index]
                            #更新输出层权重
                            self.output_emb[non_leaf_node_index] += g * nue1
                    elif self.hs:
                        codelen = word_unit.codelen
                        for d in range(codelen):
                            f = 0
                            non_leaf_node_index = word_unit.hs_path_node[d]
                            f += np.matmul(nue1, self.output_emb[non_leaf_node_index])

                            if f <= self.config.MAX_EXP:
                                g = self.lr * (1 - word_unit.hs_path_code[d])
                            elif f >= self.config.MAX_EXP:
                                g = self.lr * (1 - word_unit.hs_path_code[d] - 1)
                            else:
                                f = self.expTable[int((f + self.config.MAX_EXP) * (self.config.EXP_TABLE_SIZE / self.config.MAX_EXP / 2))]
                                g = self.lr * (1 - word_unit.hs_path_code[d] - f)
                            # 计算累计误差
                            nue1e += g * self.output_emb[non_leaf_node_index]
                            # 更新输出层权重
                            self.output_emb[non_leaf_node_index] += g * nue1

                    # 更新每次context词的向量
                    for a in range(b, self.window * 2 + 1 - b):
                        if a != self.window:
                            c = sentence_position - self.window + a
                            if c < 0:
                                continue
                            if c >= sentence_length:
                                continue
                            context_word = self.vocab_index[sentence[c]]
                            self.word_emb[context_word] += nue1e
    def train_SkipGram(self):
        for sentence in self.train_file:
            sentence_length = len(sentence)
            for sentence_position, word in enumerate(sentence):
                nue1 = np.zeros(shape=(self.config.emb_size), dtype=float)
                nue1e = np.zeros(shape=(self.config.emb_size), dtype=float)

    def initUnigramTable(self, table_size, vocab_size):
        power = 0.75
        train_words_pow = 0.0
        table = np.zeros(shape=(table_size), dtype=int)
        for a in range(vocab_size):
            train_words_pow += math.pow(self.vocab_word[a][-1], power)

        i = 0
        d1 = math.pow(self.vocab_word[i][-1], power) / train_words_pow
        for a in range(table_size):
            table[a] = i
            if a / float(table_size) > d1:
                i += 1
                d1 += math.pow(self.vocab_word[i][-1], power) / train_words_pow

            if i >= vocab_size:
                i = vocab_size - 1

        return table

    def initExpTable(self, EXP_TABLE_SIZE, MAX_EXP):
        # pre compute the epx() table
        expTable = np.empty((EXP_TABLE_SIZE + 1), dtype=float)
        for i in range(
                EXP_TABLE_SIZE + 1):  # i should be [0,EXP_TABLE_SIZE], but source code set i to [0,EXP_TABLE_SIZE)
            expTable[i] = np.exp((i / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
            expTable[i] = expTable[i] / (expTable[i] + 1)
        return expTable
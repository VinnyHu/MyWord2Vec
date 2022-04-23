import numpy as np


class WordUnit:
    # 定义一个word的信息
    def __init__(self, max_code_length):
        self.hs_path_code = [0 for i in range(max_code_length)]
        self.hs_path_node = [0 for i in range(max_code_length)]
        self.word = ''
        self.codelen = 0

class Vocab:
    # 从数据中获取词典
    # 提供2种方式，一种是从训练语料中读取；一种是从给定的词典中读取
    def __init__(self, config):
        self.config = config
        self.file = []

        self.train_file = config.vocab_file if config.learnFromVocab else config.train_file

        with open(self.train_file) as f:
            for item in f.readlines():
                self.file.append(item.strip())

        self.vocab_word = []
        self.vocab_index = {}

        self.readVocab()
        self.createBinaryTree()

    def readVocab(self):
        vocab = {}
        for sentence in self.file:
            for word in sentence.split(' '):
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        for k, v in vocab.items():
            if v >= self.config.min_count:
                current_word = WordUnit(max_code_length=self.config.max_code_length)
                current_word.word = k
                self.vocab_word.append((current_word, v))

        self.vocab_word.sort(key= lambda tup: tup[-1], reverse=True)

        index = 0
        for item in self.vocab_word:
            self.vocab_index[item.word] = index
            index += 1

    def createBinaryTree(self):
        # 因为huffman tree 是从叶子节点 向 根节点来建立的，也即自底向上。 而训练hierarchical softmax是从根向底而来的，所以需要临时数组保存下路径信息，然后再reverse过来。
        # Huffman tree 节点路径信息主要包含2个内容，一个是此节点是父节点的左儿子，还是右儿子。 一个是此节点的父节点编号。

        vocab_size = self.getVocabSize()

        count = [1e15 for i in range(2 * vocab_size + 1)]
        binary = [0 for i in range(2 * vocab_size + 1)]
        parent_node = [0 for i in range(2 * vocab_size + 1)]
        code = [0 for i in range(self.config.max_code_length)]
        point = [0 for i in range(self.config.max_code_length)]

        for a in range(vocab_size):
            count[a] = self.vocab_word[a][-1]

        # 构建Huffman tree
        left = vocab_size - 1
        right = vocab_size
        min1 = 0
        min2 = 0
        for cur_index in range(vocab_size - 1):
            if left >= 0:
                if count[left] < count[right]:
                    min1 = left
                    left -= 1
                else:
                    min1 = right
                    right += 1
            else:
                min1 = right
                right += 1

            if left >= 0:
                if count[left] < count[right]:
                    min2 = left
                    left -= 1
                else:
                    min2 = right
                    right += 1
            else:
                min2 = right
                right += 1

            count[vocab_size + cur_index] = count[min1] + count[min2]
            parent_node[min1] = vocab_size + cur_index
            parent_node[min2] = vocab_size + cur_index
            binary[min2] = 1

        # 将node的两类信息，记录进每个单词中
        for cur_index in range(vocab_size):

            b = cur_index
            num = 0 # 记录Huffman的长度

            while True:
                code[num] = count[b]
                point[num] = b
                num += 1
                b = parent_node[b]

                if b == 2 * vocab_size - 2:
                    break

            self.vocab_word[cur_index][0].codelen = num
            self.vocab_word[cur_index][0].hs_path_node = vocab_size - 2

            # 倒序
            for b in range(num):
                self.vocab_word[b][0].hs_path_code[num - b - 1] = code[b]
                self.vocab_word[b][0].hs_path_node[num - b] = point[b] - vocab_size

    def getVocabSize(self):
        return len(self.vocab_word)

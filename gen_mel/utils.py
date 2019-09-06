import numpy as np
from random import shuffle

class Lang:
    def __init__(self):
        self.item2index = {"_PAD_": 0, "_UNK_": 1, "_EOS_":2, "_START_":3}
        self.item2count = {}
        self.index2item = {0: "_PAD_", 1: "_UNK_", 2: "_EOS_", 3: "_START_"}
        self.n_items = 4

    def addSentence(self, sentence):
        for term in sentence:
            self.addItem(term)
            
    def addItem(self, item):
        if item not in self.item2index:
            self.item2index[item] = self.n_items
            self.item2count[item] = 1
            self.index2item[self.n_items] = item
            self.n_items += 1
        else:
            self.item2count[item] += 1

class PairLoader:
    def __init__(self, en_file, de_file, encode_vocab, decode_vocab):
        self.en_item2index = encode_vocab.item2index
        self.en_index2item = encode_vocab.index2item
        
        self.de_item2index = decode_vocab.item2index
        self.de_index2item = decode_vocab.index2item
        
        self.en_fp = open(en_file, 'r', encoding='utf-8')
        self.de_fp = open(de_file, 'r', encoding='utf-8')
        self.finished = False

        self.preload = 65536
        self.cahce_sents = [[], []] #0 for encode_senteces, 1 for decode_senteces
        self.cache_size = 0
            
    def reset(self):
        self.en_fp.seek(0)
        self.de_fp.seek(0)
        self.finished = False

    def read(self, Shuffling = True):
        if self.cache_size <= 0:
            self.cahce_sents = [[], []]
            for i in range(self.preload):
                en_sent = self.en_fp.readline()
                de_sent = self.de_fp.readline()
                                
                if not en_sent or not de_sent:
                    self.finished = True
                    break
                    
                if en_sent.strip() == '' or de_sent.strip() == '': continue
                    
                self.cahce_sents[0].append(en_sent)
                self.cahce_sents[1].append(de_sent)
                self.cache_size += 1

            if self.cache_size > 0 and Shuffling:
                temp = list(zip(self.cahce_sents[0], self.cahce_sents[1]))
                shuffle(temp)
                self.cahce_sents[0], self.cahce_sents[1] = zip(*temp)
                self.cahce_sents[0], self.cahce_sents[1] = list(self.cahce_sents[0]), list(self.cahce_sents[1])
                
        if self.finished and self.cache_size <= 0:
            return None, None
        else:
            self.cache_size -= 1
            return self.cahce_sents[0].pop(), self.cahce_sents[1].pop()
        
    def get_idx_list(self, sentence, item2index):
        terms_list = [term for term in sentence.split(' ') if term.strip() != '']
        idx_list = []
        for term in terms_list:
            if item2index.get(term) is not None:
                idx_list.append(item2index[term])
            else:
                idx_list.append(item2index['_UNK_'])
                
        return idx_list
        
    def gen_pairs(self, sequence_num = 5):
        while True:
            en_seq_list = []
            de_seq_list = []
            en_max_len = -1
            de_max_len = -1
            for i in range(sequence_num):
                en_sent, de_sent = self.read()
                if en_sent is None or de_sent is None: break
                
                en_sent = en_sent.strip()
                de_sent = de_sent.strip()
                
                en_idx_list = self.get_idx_list(en_sent, self.en_item2index)
                en_list_len = len(en_idx_list)
                
                de_idx_list = self.get_idx_list(de_sent, self.de_item2index)
                de_list_len = len(de_idx_list)
                
                if en_list_len > en_max_len: en_max_len = en_list_len
                if de_list_len > de_max_len: de_max_len = de_list_len
                
                en_seq_list.append((en_idx_list, en_list_len))
                de_seq_list.append((de_idx_list, de_list_len))

            if len(en_seq_list) < sequence_num or len(de_seq_list) < sequence_num:
                break
            
            # encoder data pre-process
            en_max_len += 1 #for _EOS_
            en_seq = [] #[[1,2],[2],[1]]
            en_seq_len = [] #[2,1,1]

            for i in range(len(en_seq_list)):
                en_idx_list, en_list_len = en_seq_list[i]
                en_idx_list.append(self.en_item2index['_EOS_'])
                en_list_len += 1
                
                #padding
                for j in range(en_list_len, en_max_len):
                    en_idx_list.append(self.en_item2index['_PAD_'])
                    
                en_seq.append(en_idx_list)
                en_seq_len.append(en_list_len)

            en_seq = np.array(en_seq)
            en_seq_len = np.array(en_seq_len)
            
            # decoder data pre-process
            de_max_len += 2 #for _START_ and _EOS_
            de_seq = [] #[[1,2],[2],[1]]
            de_seq_len = [] #[2,1,1]

            for i in range(len(de_seq_list)):
                de_idx_list, de_list_len = de_seq_list[i]
                de_idx_list.insert(0, self.de_item2index['_START_'])
                de_idx_list.append(self.de_item2index['_EOS_'])
                de_list_len += 2
                
                #padding
                for j in range(de_list_len, de_max_len):
                    de_idx_list.append(self.de_item2index['_PAD_'])
                    
                de_seq.append(de_idx_list)
                de_seq_len.append(de_list_len)

            de_seq = np.array(de_seq)
            de_seq_len = np.array(de_seq_len)
            
            yield en_seq, en_seq_len, de_seq, de_seq_len

class Transfrom:
    def __init__(self, encode_vocab):
        self.en_item2index = encode_vocab.item2index
        self.en_index2item = encode_vocab.index2item
        
    def get_idx_list(self, sentence, item2index):
        terms_list = [term for term in sentence.split(' ') if term.strip() != '']
        idx_list = []
        for term in terms_list:
            if item2index.get(term) is not None:
                idx_list.append(item2index[term])
            else:
                idx_list.append(item2index['_UNK_'])
                
        return idx_list
    
    def trans_input(self, sentence):
        en_seq = []
        en_seq_len = []
        
        en_idx_list = self.get_idx_list(sentence, self.en_item2index)
        en_list_len = len(en_idx_list) + 1
        en_idx_list.append(self.en_item2index['_EOS_'])
        
        en_seq.append(en_idx_list)
        en_seq_len.append(en_list_len)
        
        en_seq = np.array(en_seq)
        en_seq_len = np.array(en_seq_len)
        
        return en_seq, en_seq_len
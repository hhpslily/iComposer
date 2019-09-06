import numpy as np
import pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import Lang
from utils import Transfrom

from seq2seq import Encoder
from seq2seq import Decoder

torch.manual_seed(1)
torch.set_num_threads(4)
    
def Load_Vocab(file):
    with open(file, 'rb') as fd:
        _vocab = pickle.load(fd)
    return _vocab
    
def Load_Parameters(file):
    with open(file, 'rb') as fd:
        parameters_dict = pickle.load(fd)
    return parameters_dict
    
if __name__ == '__main__':

    en_vocab_file = 'en_vocab.pkl'
    de_vocab_file = 'de_vocab.pkl'
    encoder_model_file = 'encoder_ly.81.pt'
    decoder_model_file = 'decoder_ly.81.pt'
    hyper_parameters_file = 'parameters_dict.pkl'
    
    en_vocab = Load_Vocab(en_vocab_file)
    de_vocab = Load_Vocab(de_vocab_file)
    
    trf = Transfrom(en_vocab)
    
    parameters_dict = Load_Parameters(hyper_parameters_file)

    en_embedding_dim = parameters_dict['en_embedding_dim']
    de_embedding_dim = parameters_dict['de_embedding_dim']
    hidden_dim = parameters_dict['hidden_dim']
    num_layers = parameters_dict['num_layers']
    bidirectional = parameters_dict['bidirectional']
    use_lstm = parameters_dict['use_lstm']
    use_cuda = False
    batch_size = 1
    dropout_p = 0.0
    
    encoder = Encoder(en_embedding_dim, hidden_dim, en_vocab.n_items, num_layers, dropout_p, bidirectional, use_lstm, use_cuda)
    decoder = Decoder(de_embedding_dim, hidden_dim, de_vocab.n_items, num_layers, dropout_p, bidirectional, use_lstm, use_cuda)
    
    encoder.load_state_dict(torch.load(encoder_model_file, map_location='cpu'))
    decoder.load_state_dict(torch.load(decoder_model_file, map_location='cpu'))
        
    encoder.eval()
    decoder.eval()

    f_en_test = open('data/de_test.txt', 'r', encoding = 'utf-8')
    f_de_pred = open('en_pred.txt', 'w', encoding = 'utf-8')
    
    
    while True:
        en_sent = f_en_test.readline()

        if not en_sent: break
        
        sent = en_sent.strip()
        en_seq, en_seq_len = trf.trans_input(sent)

        en_seq = torch.LongTensor(en_seq)
        encoder_input = en_seq
        encoder_output, encoder_state = encoder(encoder_input, en_seq_len)
        
        # initial decoder hidden
        decoder_state = decoder.init_state(encoder_state)

        # Start decoding
        decoder_inputs = torch.LongTensor([de_vocab.item2index['_START_']])

        pred_char = ''
        pred_sent = []
        if use_cuda: decoder_inputs = decoder_inputs.cuda()
        decoder_outputs, decoder_state = decoder(decoder_inputs, encoder_output, decoder_state)

        max_len = len(en_sent.split())
        
        # Greedy search
        while pred_char != '_EOS_':
            log_prob, v_idx = decoder_outputs.data.topk(1)
            pred_char = de_vocab.index2item[v_idx.item()]
            pred_sent.append(pred_char)

            if(len(pred_sent) > max_len): break
            
            decoder_inputs = torch.LongTensor([v_idx.item()])
            if use_cuda: decoder_inputs = decoder_inputs.cuda()
            decoder_outputs, decoder_state = decoder(decoder_inputs, encoder_output, decoder_state)
        
        pred_list = []
        
        scoring_stop = False
        for i in range(max_len):
            if not scoring_stop:
                if pred_sent[i] == '_EOS_':
                    scoring_stop = True
                if pred_sent[i] != '_EOS_':
                    pred_list.append(pred_sent[i])

        f_de_pred.write(' '.join(pred_list) + '\n')

    f_de_pred.close()
    

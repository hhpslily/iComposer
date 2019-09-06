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
    encoder_model_file = 'encoder.5.pt'
    decoder_model_file = 'decoder.5.pt'
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

    f_en_test = open('en_test.txt', 'r', encoding = 'utf-8')
    f_de_test = open('de_test.txt', 'r', encoding = 'utf-8')
    f_de_pred = open('de_pred.txt', 'w', encoding = 'utf-8')
    
    correct = 0
    total_token = 0
    
    while True:
        en_sent = f_en_test.readline()
        de_sent = f_de_test.readline()

        if not en_sent or not de_sent: break
        
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

        de_sent = de_sent.strip().split(' ')
        max_len = len(de_sent)

        
        # Greedy search
        while True:
            log_prob, v_idx = decoder_outputs.data.topk(1)
            pred_char = de_vocab.index2item[v_idx.item()]
            pred_sent.append(pred_char)

            if(len(pred_sent) > max_len): break
            
            decoder_inputs = torch.LongTensor([v_idx.item()])
            if use_cuda: decoder_inputs = decoder_inputs.cuda()
            decoder_outputs, decoder_state = decoder(decoder_inputs, encoder_output, decoder_state)
        
        '''
        # Beam search
        samples = []
        topk = 5
        log_prob, v_idx = decoder_outputs.data.topk(topk)
        for k in range(topk):
            samples.append([[v_idx[0][k].item()], log_prob[0][k], decoder_state])

        for _ in range(max_len):
            new_samples = []
            
            for sample in samples:
                v_list, score, decoder_state = sample
     
                if v_list[-1] == de_vocab.item2index['_EOS_']:
                    new_samples.append([v_list, score, decoder_state])
                    continue
                
                decoder_inputs = torch.LongTensor([v_list[-1]])

                decoder_outputs, new_states = decoder(decoder_inputs, encoder_output, decoder_state)
                log_prob, v_idx = decoder_outputs.data.topk(topk)
                
                for k in range(topk):
                    new_v_list = []
                    new_v_list += v_list + [v_idx[0][k].item()]
                    new_samples.append([new_v_list, score + log_prob[0][k], new_states])

            new_samples = sorted(new_samples, key = lambda sample: sample[1], reverse=True)
            samples = new_samples[:topk]

        best_score = -(1e8)
        best_idx = -1
        best_states = None
        for i, sample in enumerate(samples):
            v_list, score, states = sample
            if score.item() > best_score:
                best_score = score
                best_idx = i
                best_states = states

        v_list, score, states = samples[best_idx]
        for v_idx in v_list:
            pred_sent.append(de_vocab.index2item[v_idx])
        '''

        pred_list = []
        
        scoring_stop = False
        for i in range(max_len):
            if not scoring_stop:
                if pred_sent[i] == '_EOS_':
                    scoring_stop = True
                elif pred_sent[i] == de_sent[i]:
                    correct += 1

                if pred_sent[i] != '_EOS_': pred_list.append(pred_sent[i])
            total_token += 1

        f_de_pred.write(' '.join(pred_list) + '\n')
            
    print(correct / total_token)
    

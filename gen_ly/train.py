import numpy as np
import pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import Lang
from utils import PairLoader

from seq2seq import Encoder
from seq2seq import Decoder

torch.manual_seed(1)
torch.set_num_threads(4)

def build_vocab(file, n_limit = 1):
    vocab = Lang()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '': continue
            vocab.addSentence(line.split(' '))

    new_vocab = Lang()
    for i in range(4, vocab.n_items):
        term = vocab.index2item[i]
        freq = vocab.item2count[term]
        if freq < n_limit: continue
        new_vocab.addItem(term)
        new_vocab.item2count[term] = freq
        
    return new_vocab

def mask_select(ones_maxtrix, inputs, targets, outputs):

    seq_num = inputs.size()
    nonzero_seq, _ = torch.nonzero(inputs).size()

    if seq_num[0] == nonzero_seq:
        return outputs, targets

    seq_num, vocab_size = outputs.size()

    if use_cuda: temp = torch.mm(inputs.float().cuda().view(-1, 1), ones_matrix)
    else: temp = torch.mm(inputs.float().view(-1, 1), ones_matrix)

    o_mask = temp.ge(1)
    outputs = torch.masked_select(outputs, o_mask).view(-1, vocab_size)

    t_mask = inputs.ge(1)
    targets = torch.masked_select(targets, t_mask)
    
    return outputs, targets

if __name__ == '__main__':

    en_file = 'de_train.txt'
    de_file = 'en_train.txt'

    en_vocab = build_vocab(en_file)
    de_vocab = build_vocab(de_file)

    with open('en_vocab.pkl', 'wb') as fd:
        pickle.dump(en_vocab, fd)
    with open('de_vocab.pkl', 'wb') as fd:
        pickle.dump(de_vocab, fd)
    
    pl = PairLoader(en_file, de_file, en_vocab, de_vocab)
    
    en_embedding_dim = 100
    de_embedding_dim = 50
    hidden_dim = 200
    num_layers = 1
    batch_size = 4
    dropout_p = 0.0
    
    bidirectional = True
    use_cuda = False 
    use_lstm = True
    
    parameters_dict = {}
    parameters_dict['en_embedding_dim'] = en_embedding_dim
    parameters_dict['de_embedding_dim'] = de_embedding_dim
    parameters_dict['hidden_dim'] = hidden_dim
    parameters_dict['num_layers'] = num_layers
    parameters_dict['bidirectional'] = bidirectional
    parameters_dict['use_lstm'] = use_lstm

    with open('parameters_dict.pkl', 'wb') as fd:
        pickle.dump(parameters_dict, fd)
    
    batch_total = sum([1 for _ in pl.gen_pairs(batch_size)])
    ones_matrix = autograd.Variable(torch.ones(1, de_vocab.n_items))
    
    encoder = Encoder(en_embedding_dim, hidden_dim, en_vocab.n_items, num_layers, dropout_p, bidirectional, use_lstm, use_cuda)
    decoder = Decoder(de_embedding_dim, hidden_dim, de_vocab.n_items, num_layers, dropout_p, bidirectional, use_lstm, use_cuda)

    
    encoder_model_file = 'encoder_rev.7.pt'
    decoder_model_file = 'decoder_rev.7.pt'
    encoder.load_state_dict(torch.load(encoder_model_file))
    decoder.load_state_dict(torch.load(decoder_model_file))

    '''
    #Load Pre-trained Embedding
    model_file = 'bi_gru.100.100.2.pt'
    if model_file != '' : model.load_state_dict(torch.load(model_file))
    else: model.load_pre_train_emb('cityu_training.char.emb.npy', 'cityu_training.char.dict', vocab)
    '''
    
    loss_function = nn.NLLLoss(reduction = 'sum', ignore_index = de_vocab.item2index['_PAD_'])
    en_optimizer = optim.Adam(encoder.parameters(), lr = 1e-3, weight_decay = 0)
    de_optimizer = optim.Adam(decoder.parameters(), lr = 1e-3, weight_decay = 0)
    
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        ones_matrix = ones_matrix.cuda()
        loss_function = loss_function.cuda()
        
    for epoch in range(20):
        
        pl.reset()
        encoder.train()
        decoder.train()
        total_loss = torch.Tensor([0])
        total_token = 0

        for batch_idx, (en_seq, en_seq_len, de_seq, de_seq_len) in enumerate(pl.gen_pairs(batch_size)):

            en_optimizer.zero_grad()
            de_optimizer.zero_grad()

            en_seq = torch.LongTensor(en_seq)
            de_seq = torch.LongTensor(de_seq)
            if use_cuda:
                en_seq = en_seq.cuda()
                de_seq = de_seq.cuda()

            # Encode
            encoder_input = en_seq
            encoder_output, encoder_state = encoder(encoder_input, en_seq_len)
            
            # initial decoder hidden
            decoder_state = decoder.init_state(encoder_state)

            # Decode
            batch_size, max_len = de_seq.size()
            words_num = np.sum(de_seq_len)
            total_token += words_num
            loss = 0
            
            for i in range(max_len - 1):
                decoder_inputs = de_seq[:,i]
                decoder_targets = de_seq[:,i + 1]

                decoder_outputs, decoder_state = decoder(decoder_inputs, encoder_output, decoder_state)
                #outputs, targets = mask_select(ones_matrix, decoder_inputs, decoder_targets, decoder_outputs)

                loss += loss_function(decoder_outputs, decoder_targets)

            batch_loss = loss.item() / words_num
            print('Epoch: {}, Batch: {}\{}, Batch Loss: {}'.format(epoch + 1, batch_idx + 1, batch_total, batch_loss))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)
            en_optimizer.step()
            de_optimizer.step()
            
            total_loss += loss.item()

        total_loss = total_loss.item() / total_token
        print('Epoch: {}, Total Loss: {}'.format(epoch + 1, total_loss))
        
        ''' save model '''
        torch.save(encoder.state_dict(), 'encoder_rev.' + str(epoch + 1) + '.pt')
        torch.save(decoder.state_dict(), 'decoder_rev.' + str(epoch + 1) + '.pt')


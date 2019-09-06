import numpy as np
import pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parameter import Parameter

torch.manual_seed(1)
torch.set_num_threads(4)

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers = 1, dropout_p = 0.0, bidirectional = False, use_lstm = False, use_cuda = False):
        super(RNN, self).__init__()
        self.use_lstm = use_lstm
        self.use_cuda = use_cuda
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        
        if self.use_lstm: self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout = dropout_p, bidirectional=self.bidirectional, batch_first=True)
        else: self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout = dropout_p, bidirectional=self.bidirectional, batch_first=True)
        
        for i in range(len(self.rnn.all_weights)):
            for j in range(len(self.rnn.all_weights[i])):
                if j < 2:
                    nn.init.xavier_normal_(self.rnn.all_weights[i][j])
                else:
                    self.rnn.all_weights[i][j].data.fill_(0)

    def forward(self, seq, seq_len, embeddings, previous_state = None):
        
        """sort"""
        if seq_len is not None:
            seq_sort_idx = np.argsort(-seq_len)
            seq_unsort_idx = torch.LongTensor(np.argsort(seq_sort_idx))
            seq_len = seq_len[seq_sort_idx]
            seq = autograd.Variable(seq[torch.LongTensor(seq_sort_idx)])
            
            if self.use_cuda:
                seq_unsort_idx = seq_unsort_idx.cuda()
                seq = seq.cuda()
                
            """pack"""
            seq_emb = embeddings(seq)
            seq_emb_p = torch.nn.utils.rnn.pack_padded_sequence(seq_emb, seq_len, batch_first=True)
            
            """process using RNN"""
            if self.use_lstm: out_pack, (ht, ct) = self.rnn(seq_emb_p, previous_state)
            else: out_pack, ht = self.rnn(seq_emb_p, previous_state)
            
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)# (sequence, lengths)
            out = out[0]
            out = out[seq_unsort_idx] # (batch, max_lengths, hidden_size)

            ht = torch.transpose(ht, 0, 1)[seq_unsort_idx] 
            ht = torch.transpose(ht, 0, 1) 

            if self.use_lstm:
                ct = torch.transpose(ct, 0, 1)[seq_unsort_idx]
                ct = torch.transpose(ct, 0, 1) 
                state = (ht, ct)
            else:
                state = ht
        else:
            if self.use_cuda: seq = seq.cuda()
            seq = autograd.Variable(seq)
            seq_emb = embeddings(seq)
            seq_emb = torch.unsqueeze(seq_emb, 1)
            out, state = self.rnn(seq_emb, previous_state)
            
        return out, state

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers = 1, dropout_p = 0.0, bidirectional = False, use_lstm = False, use_cuda = False):
        super(Encoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.use_cuda = use_cuda
        self.use_lstm = use_lstm
        
        # The embedding layer 
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # The rnn layer
        self.rnn = RNN(self.embedding_dim, self.hidden_dim, self.num_layers, self.dropout_p, self.bidirectional, self.use_lstm, self.use_cuda)
        
        # initial
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, seq, seq_len):
        out, state = self.rnn(seq, seq_len, self.embeddings)
        return out, state
    
    def load_pre_train_emb(self, emb_file, dict_file, vocab):
        import pickle
        
        pre_trained_embedding = np.load(emb_file)
        with open(dict_file, 'rb') as fd:
            pre_trained_embedding_dict = pickle.load(fd)
        
        for term in pre_trained_embedding_dict.keys():
            if term in vocab.item2index:
                index = pre_trained_embedding_dict[term]
                norm = np.linalg.norm(pre_trained_embedding[index])
                pre_trained_embedding[index] /= norm
                emb = torch.from_numpy(pre_trained_embedding[index])
                self.embeddings.weight[vocab.item2index[term]].data.copy_(emb)

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers = 1, dropout_p = 0.0, bidirectional = False, use_lstm = False, use_cuda = False):
        super(Decoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.bidirectional_encoder = bidirectional
        self.use_cuda = use_cuda
        self.use_lstm = use_lstm
        
        if self.bidirectional_encoder: self.hidden_dim *= 2
            
        # The embedding layer 
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # The rnn layer 
        self.rnn = RNN(self.embedding_dim, self.hidden_dim, self.num_layers, self.dropout_p, False, self.use_lstm, self.use_cuda)
        
        self.linear1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        # initial
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.linear1.bias.data.fill_(0)
        self.linear2.bias.data.fill_(0)
    
    def init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
        
    def forward(self, seq, encoder_output, previous_state):
        
        rnn_out, state = self.rnn(seq, None, self.embeddings, previous_state)
        batch_size, _, hidden_dim = rnn_out.size()
        
        # dot attetion
        attn_weights = torch.bmm(encoder_output, rnn_out.transpose(1, 2))
        attn_weights = self.softmax(attn_weights.transpose(1, 2))
        
        context_embeddings = torch.bmm(attn_weights, encoder_output)
        
        combine = torch.cat((rnn_out, context_embeddings), 2)
        combine = torch.tanh(self.linear1(combine))

        vocab_space = self.linear2(combine.view(batch_size, -1))
        output = self.log_softmax(vocab_space)
        
        return output, state
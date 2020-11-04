import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import random
import math
import numpy as np






class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, z_dim, n_layers, dropout, bidirectional=False, teacher_force=0.5, rnn_type='lstm', z_mode=None, setting=None, device=None):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.z_mode = z_mode
        self.device = device
        self.setting = setting
        self.layer_dim = n_layers*2 if bidirectional else n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim+z_dim, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim+z_dim, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim+z_dim, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidirectional)
        self.z2h_c = nn.Linear(z_dim, hid_dim, bias=False)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.teacher_force = teacher_force


    def forward(self, enc_z, sen):
        
        if self.rnn_type == 'lstm':
            b_size = enc_z.shape[0]
            enc_z = torch.transpose(enc_z.view(b_size, self.layer_dim, -1), 0, 1) #[n layers * n directions, batch size, z_dim]
       
            hidden = self.z2h_c(enc_z) # [n layers * n directions, batch size, hid_dim]
            cell = hidden[:]
            
        elif self.rnn_type == 'rnn' or self.rnn_type == 'gru':
            b_size = enc_z.shape[0]
            enc_z = torch.transpose(enc_z.view(b_size, self.layer_dim, -1), 0, 1) #[n layers * n directions, batch size, z_dim]
            
            hidden = self.z2h_c(enc_z) # [n layers * n directions, batch size, hid_dim]
        

        embedded = self.dropout(self.embedding(sen[:-1,:]))

        z_expand = enc_z.expand(embedded.shape[0],embedded.shape[1], enc_z.shape[2])

        dec_input = torch.cat([embedded,z_expand],2)

        if self.rnn_type == 'lstm':
            output, _ = self.rnn(dec_input, (hidden, cell))
            prediction = self.out(output) # [seq_sen, batch, vec_dim]
        elif self.rnn_type == 'rnn' or 'gru':
            output, _ = self.rnn(dec_input, hidden)
            prediction = self.out(output) # [seq_sen, batch, vec_dim]

        return prediction

    def singledecode(self, input, hidden, cell, lat_z=None):
        # first input to the decoder is the <sos> tokens
        input = input.unsqueeze(0)
        
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        
        emba_cat = torch.cat([embedded,lat_z], dim=2)

        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(emba_cat, (hidden, cell))
        elif self.rnn_type == 'rnn' or self.rnn_type == 'gru':
            output, hidden = self.rnn(emba_cat, hidden)
        
        prediction = self.out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

    def loss(self, prod, target, weight):
        
        recon_loss = F.cross_entropy(
            prod.view(-1, prod.shape[2]), target[1:].view(-1),
            ignore_index=0, reduction="sum")
        return recon_loss

    
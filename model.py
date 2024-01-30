# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch    
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import typing as tp
import math

class Summarization(nn.Module):
    # Multi-View Summarization Module 
    def __init__(self, embed_size, smry_k):
        super(Summarization, self).__init__()
        # dilation conv
        out_c = [256,256,256]#, 128, 128, 128]
        k_size = [2, 3, 4]#, 5, 5, 5]
        #dila = [2,3,4]#, 1, 2, 3]
        #pads = [0, 1, 2, 3]#, 2, 4, 6]
        convs_dilate = [nn.Conv1d(embed_size, out_c[i], k_size[i], padding="same") \
                        for i in range(len(out_c))]
        self.convs_dilate = nn.ModuleList(convs_dilate)
        self.convs_fc = nn.Linear(256, smry_k)
        
    def forward(self, rgn_emb):
        x = rgn_emb.transpose(1, 2)    #(bs, dim, num_r)
        x = [torch.matmul(F.softmax(self.convs_fc((F.relu(conv(x))).transpose(1, 2)),dim=1).transpose(1, 2), rgn_emb) for conv in self.convs_dilate]
        x = torch.cat(x, dim=1) 

        return x

class MultiView(nn.Module):

    def __init__(self, embed_size, smry_k):
        super(MultiView, self).__init__()
        # dilation conv
        out_c = [256, 128, 128, 64, 64, 64, 64]
        k_size = [1, 3, 3, 3, 5, 5, 5]
        dila = [1, 1, 2, 3, 1, 2, 3]
        pads = [0, 1, 2, 3, 2, 4, 6]
        convs_dilate = [nn.Conv1d(embed_size, out_c[i], k_size[i], dilation=dila[i], padding=pads[i]) \
                        for i in range(len(out_c))]
        self.convs_dilate = nn.ModuleList(convs_dilate)
        self.convs_fc = nn.Linear(768, smry_k)

    def forward(self, rgn_emb):
        x = rgn_emb.transpose(1, 2)    #(bs, dim, num_r)
        x = [F.relu(conv(x)) for conv in self.convs_dilate]
        x = torch.cat(x, dim=1) #(bs, 1024, num_r)
        x = x.transpose(1, 2)   #(bs, num_r, 1024)
        smry_mat = self.convs_fc(x)    #(bs, num_r, k)
        
        return smry_mat    
    
    
def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features#, pool_weights

    def get_pe(self, length):
        """
        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe

class Model(nn.Module):   
    def __init__(self, encoder, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.gpool = GPO(8, 8)

        self.mvs = MultiView(768,256)
        self.args = args
        
    def forward(self, code_inputs=None, nl_inputs=None): 

        if code_inputs is not None:
            
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            #Multi-View Summarization

            ##########token-level
            smry_mat = self.mvs(outputs)
            L = F.softmax(smry_mat, dim=1)
            outputs = torch.matmul(L.transpose(1, 2), outputs) #(bs, k, dim)
            ##########token-level
            
            vect = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            #vect = self.gpool(outputs,(torch.full([outputs.size(0)],outputs.size(1))).to(self.args.device))
            vect = torch.nn.functional.normalize(vect, p=2, dim=1) 

            return vect
        

        elif nl_inputs is not None:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            vect = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            #vect = self.gpool(outputs,(torch.full([outputs.size(0)],outputs.size(1))).to(self.args.device))
            vect = torch.nn.functional.normalize(vect, p=2, dim=1) 

            return vect

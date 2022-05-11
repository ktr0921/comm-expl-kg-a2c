import spacy
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StateNetwork(nn.Module):
    def __init__(self, gat_emb_size, vocab, embedding_size, hidden_size, dropout, tsv_file, embeddings=None):
        super(StateNetwork, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.gat_emb_size = gat_emb_size
        self.hidden_size = hidden_size
        #self.params = params
        self.gat = GAT(gat_emb_size, 3, dropout, alpha=0.2, nheads=1)
        # dropout, alpha, nheads

        self.pretrained_embeds = nn.Embedding(self.vocab_size, self.embedding_size)
        self.vocab_kge = self.load_vocab_kge(tsv_file)
        #self.init_state_ent_emb(params['embedding_size'])
        self.state_ent_emb = nn.Embedding.from_pretrained(torch.zeros((len(self.vocab_kge), self.gat_emb_size)), freeze=False)
        self.fc1 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, self.hidden_size)

    def init_state_ent_emb(self, emb_size):
        embeddings = torch.zeros((len(self.vocab_kge), emb_size))
        for i in range(len(self.vocab_kge)):
            graph_node_text = self.vocab_kge[i].split('_')
            graph_node_ids = []
            for w in graph_node_text:
                if w in self.vocab.keys():
                    if self.vocab[w] < len(self.vocab) - 2:
                        graph_node_ids.append(self.vocab[w])
                    else:
                        graph_node_ids.append(1)
                else:
                    graph_node_ids.append(1)
            graph_node_ids = torch.LongTensor(graph_node_ids).to(device)
            cur_embeds = self.pretrained_embeds(graph_node_ids)

            cur_embeds = cur_embeds.mean(dim=0)
            embeddings[i, :] = cur_embeds
        self.state_ent_emb = nn.Embedding.from_pretrained(embeddings, freeze=False)

    def load_vocab_kge(self, tsv_file):
        ent = {}
        with open(tsv_file, 'r') as f:
            for line in f:
                e, eid = line.split('\t')
                ent[int(eid.strip())] = e.strip()
        return ent

    def forward(self, graph_rep):
        out = []
        for g in graph_rep:
            node_feats, adj = g
            adj = torch.IntTensor(adj).to(device)
            x = self.gat.forward(self.state_ent_emb.weight, adj).view(-1)
            out.append(x.unsqueeze(0))
        out = torch.cat(out)
        ret = self.fc1(out)
        return ret


class ActionDrQA(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, batch_size, recurrent=True):
        super(ActionDrQA, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.recurrent = recurrent
        self.hidden_size = hidden_size

        # embedding
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)

        # rnn
        self.enc_look  = PackedEncoderRNN(self.vocab_size, self.hidden_size)
        self.enc_inv   = PackedEncoderRNN(self.vocab_size, self.hidden_size)
        self.enc_ob    = PackedEncoderRNN(self.vocab_size, self.hidden_size)
        self.enc_preva = PackedEncoderRNN(self.vocab_size, self.hidden_size)

        # init hidden layer
        self.h_look  = self.enc_look.initHidden(self.batch_size)
        self.h_inv   = self.enc_inv.initHidden(self.batch_size)
        self.h_ob    = self.enc_ob.initHidden(self.batch_size)
        self.h_preva = self.enc_preva.initHidden(self.batch_size)

        # linear
        self.fcx = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.fch = nn.Linear(self.hidden_size * 4, self.hidden_size)

    def reset_hidden(self, done_mask_tt):
        '''
        Reset the hidden state of episodes that are done.

        :param done_mask_tt: Mask indicating which parts of hidden state should be reset.
        :type done_mask_tt: Tensor of shape [BatchSize x 1]

        '''
        self.h_look = done_mask_tt.detach() * self.h_look
        self.h_inv = done_mask_tt.detach() * self.h_inv
        self.h_ob = done_mask_tt.detach() * self.h_ob
        self.h_preva = done_mask_tt.detach() * self.h_preva

    def clone_hidden(self):
        ''' Makes a clone of hidden state. '''
        self.tmp_look = self.h_look.clone().detach()
        self.tmp_inv = self.h_inv.clone().detach()
        self.h_ob = self.h_ob.clone().detach()
        self.h_preva = self.h_preva.clone().detach()

    def restore_hidden(self):
        ''' Restores hidden state from clone made by clone_hidden. '''
        self.h_look = self.tmp_look
        self.h_inv = self.tmp_inv
        self.h_ob = self.h_ob
        self.h_preva = self.h_preva

    def forward(self, obs):
        '''
        :param obs: Encoded observation tokens.
        :type obs: np.ndarray of shape (Batch_Size x 4 x 300)

        '''
        x_l, h_l = self.enc_look(torch.LongTensor(obs[:,0,:]).to(device), self.h_look.to(device))
        x_i, h_i = self.enc_inv(torch.LongTensor(obs[:,1,:]).to(device), self.h_inv.to(device))
        x_o, h_o = self.enc_ob(torch.LongTensor(obs[:,2,:]).to(device), self.h_ob.to(device))
        x_p, h_p = self.enc_preva(torch.LongTensor(obs[:,3,:]).to(device), self.h_preva.to(device))

        if self.recurrent:
            self.h_look = h_l
            self.h_ob = h_o
            self.h_preva = h_p
            self.h_inv = h_i

        x = F.relu(self.fcx(torch.cat((x_l, x_i, x_o, x_p), dim=1)))
        h = F.relu(self.fch(torch.cat((h_l, h_i, h_o, h_p), dim=2)))

        return x, h


class EdgePredictor(nn.Module):
    def __init__(self, vocab_size, batch_size, edge_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.enc_node = PackedEncoderRNN(self.vocab_size, 100)  # rnn
        self.fc = nn.Linear(100, edge_size)                    # linear

    def forward(self, node, h_in):
        node = torch.LongTensor(node).to(device)
        h_in = self.reset_hidden(node.size(0)) if h_in is None else h_in
        y_out, h_out = self.enc_node(node, h_in)
        edge_distribution = F.relu(self.fc(y_out))

        return edge_distribution, h_out

    def reset_hidden(self, init_size=None):
        init_size = self.batch_size if init_size is None else init_size
        return self.enc_node.initHidden(init_size)


class GAT(nn.Module):
    def __init__(self, gat_emb_size, gat_hidden_size, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(gat_emb_size, gat_hidden_size, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout)
        return x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):#, hidden):
        embedded = self.embedding(input)#.unsqueeze(0)
        hidden = self.initHidden(input.size(0))
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)


class PackedEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PackedEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input).permute(1,0,2) # T x Batch x EmbDim
        if hidden is None:
            hidden = self.initHidden(input.size(0))

        lengths = []
        for n in input:
            curr_len = torch.nonzero(n)
            if curr_len.shape[0] == 0:
                lengths.append(torch.Tensor([1]))
            else:
                lengths.append(curr_len[-1] + 1)
        lengths = torch.tensor(lengths, dtype=torch.long).to(device)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False).to(device)
        output, hidden = self.gru(packed, hidden)
        # Unpack the padded sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        # Return only the last timestep of output for each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), output.size(2)).unsqueeze(0)
        output = output.gather(0, idx).squeeze(0)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        #self.combine = nn.Linear(hidden_size*2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)#, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        input = input.unsqueeze(0).expand(self.hidden_size, -1, -1)#.transpose(0, 1).contiguous()
        output, hidden = self.gru(input, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self, batch):
        return torch.zeros(1, batch, self.hidden_size).to(device)#).to(device)


class UnregDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(UnregDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.training = True
        #print(self.p)

    def forward(self, X):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            sample = binomial.sample(X.size()).to(device)
            return X * sample
        return X


class DecoderRNN2(nn.Module):
    def __init__(self, hidden_size, output_size, action_emb, graph_dropout):
        super(DecoderRNN2, self).__init__()
        self.hidden_size = hidden_size
        self.action_emb = action_emb#nn.Embedding(output_size, hidden_size)
        self.combine = nn.Linear(action_emb.embedding_dim + int(hidden_size/2), hidden_size)
        # self.combine = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.graph_dropout = UnregDropout(p=graph_dropout)#, training=False)#, inplace=True)
        self.graph_dropout_perc = graph_dropout

    def forward(self, input, hidden, encoder_output, graph_mask=None):
        output = self.action_emb(input).unsqueeze(0)
        encoder_output = encoder_output.unsqueeze(0)
        output = torch.cat((output, encoder_output), dim=-1)
        output = self.combine(output)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        ret_output = output.clone().detach()

        if self.graph_dropout_perc != 1:
            with torch.no_grad():
                norm = ret_output.norm(p=2, dim=1, keepdim=True)
                ret_output = ret_output.div(norm)
                # graph_mask = self.graph_dropout(((~graph_mask).float()))#, p=0.5, training=False)
                # graph_mask = ~(graph_mask.bool())
                ret_output[~graph_mask] = float('-inf')

        return output, ret_output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size).to(device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.2):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size)#).to(device)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    # gat_emb_size, gat_hidden_size
    def __init__(self, gat_emb_size, gat_hidden_size, dropout, alpha, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.gat_emb_size = gat_emb_size
        self.gat_hidden_size = gat_hidden_size
        self.alpha = alpha
        self.concat = concat

        torch_floattensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(torch.Tensor(gat_emb_size, gat_hidden_size).type(torch_floattensor), gain=np.sqrt(2.0)),
            requires_grad=True
        )
        self.a = nn.Parameter(
            nn.init.xavier_uniform_(torch.Tensor(2*gat_hidden_size, 1).type(torch_floattensor), gain=np.sqrt(2.0)),
            requires_grad=True
        )

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.gat_hidden_size)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = torch.zeros_like(e)
        zero_vec = zero_vec.fill_(9e-15)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.gat_emb_size) + ' -> ' + str(self.gat_hidden_size) + ')'


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, dropout_ratio, embeddings, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = embeddings#nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.gru = nn.LSTM(
            embedding_size, hidden_size, self.num_layers,
            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional
        )
        self.encoder2decoder = nn.Linear(
            hidden_size * self.num_directions, hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.to(device), c0.to(device)

    def forward(self, inputs, lengths=0):
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)

        enc_h, (enc_h_t, enc_c_t) = self.gru(embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx = self.drop(enc_h)

        #print("lstm", ctx.size(), c_t.size())

        # tmpl2obj_input, tmpl2obj_init_h, tmpl2obj_enc_oinputs
        return ctx,decoder_init,c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)






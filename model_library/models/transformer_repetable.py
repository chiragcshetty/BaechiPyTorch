# code from: https://github.com/jadore801120/attention-is-all-you-need-pytorch
# modified by cshetty2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

########## Primitives ##############################
class _matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1,x2):
        return torch.matmul(x1,x2)
    
class _addLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2

## Explained: https://jamesmccaffrey.wordpress.com/2020/09/17/an-example-of-using-the-pytorch-masked_fill-function/
class _maskfill(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        return x.masked_fill(mask == 0, -1e9)
    
class _concatenateLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dim, *x):
        return torch.cat(x, dim)
    
class _multiply(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, factor):
        return factor*x

class _divide(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, factor):
        return x/factor

class _view(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, dim):
        return t.view(*dim)

class _transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, dim):
        return t.transpose(*dim).contiguous()

class _unsqueeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, dim):
        return t.unsqueeze(dim)

class _contiguous(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return t.contiguous()
##############################################
# Constants
PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

vocab_size = 30000
seq_length = 50

len_q = seq_length
len_k = seq_length
len_v = seq_length

with_gpu_split = 0

##################### Modules #################

# Modules
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.matmul1 = _matmul()
        self.matmul2 = _matmul()
        self.maskfill = _maskfill()
        self.softmax = nn.Softmax(dim=-1)
        #self.dropout = nn.Dropout(attn_dropout)
        
    # size of q, k, v each is batch x n_head x seq_len x dv = batch x 8 x 50 x 512
    def forward(self, q, k, v, mask=None):

        attn = self.matmul1(q / self.temperature, k.transpose(2, 3)) # size of attn, mask = seq_len_src*seq_len_trg = seq_len*seq_len in our case

        if mask is not None:
            attn = self.maskfill(attn, mask)
            #attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.softmax(attn)
        #attn = self.dropout(attn)
        output = self.matmul2(attn, v)

        return output

#____________ For Unboxed_________________________  
class ScaledDotProductAttention_unboxed(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.matmul1 = _matmul()
        self.matmul2 = _matmul()
        self.maskfill = _maskfill()
        self.softmax = nn.Softmax(dim=-1)
        #self.dropout = nn.Dropout(attn_dropout)
        
    # size of q, k, v each is batch x seq_len x dv = batch x 50 x 512
    def forward(self, q, k, v, mask=None):

        attn = self.matmul1(q / self.temperature, k.transpose(1, 2)) # size of attn, mask = seq_len_src*seq_len_trg = seq_len*seq_len in our case

        if mask is not None:
            attn = self.maskfill(attn, mask)
            #attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.softmax(attn)
        #attn = self.dropout(attn)
        output = self.matmul2(attn, v)

        return output
#-------------------------------------------------------------
#SubLayers

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.view_q = _view()
        self.transpose_q = _transpose()
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.view_k = _view()
        self.transpose_k = _transpose()
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.view_v = _view()
        self.transpose_v = _transpose()

        #self.unsqueeze = _unsqueeze()
        
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.view_attn = _view()
        self.transpose_attn = _transpose()

        #self.dropout = nn.Dropout(dropout)
        self.addlayer = _addLayer()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        ## Predicatbale init
        const_w = 1/1024
        torch.nn.init.constant_(self.w_qs.weight, const_w )
        torch.nn.init.constant_(self.w_ks.weight, const_w )
        torch.nn.init.constant_(self.w_vs.weight, const_w )
        torch.nn.init.constant_(self.fc.weight, const_w )

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        #sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.view_q(self.w_qs(q), (-1, len_q, n_head, d_k))
        k = self.view_k(self.w_ks(k), (-1, len_k, n_head, d_k))
        v = self.view_v(self.w_vs(v), (-1, len_v, n_head, d_v))

        # Transpose for attention dot product: b x n x lq x dv
        q=self.transpose_q(q,(1, 2) )
        k=self.transpose_k(k,(1, 2) )
        v=self.transpose_v(v,(1, 2) )

        if mask is not None:
            #mask = self.unsqueeze(mask,1)   # For head axis broadcasting.
            mask = mask.unsqueeze(1)

        q = self.attention(q, k, v, mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        
        q = self.view_attn(self.transpose_attn(q,(1, 2)), (-1, len_q, n_head*d_v))
        q = self.fc(q) #q = self.dropout(self.fc(q))
        q = self.addlayer(q,residual)

        q = self.layer_norm(q)

        return q


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.relu = nn.ReLU(inplace=True)
        self.addlayer = _addLayer()
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        #self.dropout = nn.Dropout(dropout)

        const_w = -1/1024
        torch.nn.init.constant_(self.w_1.weight, const_w ); torch.nn.init.zeros_(self.w_1.bias)
        torch.nn.init.constant_(self.w_2.weight, const_w ); torch.nn.init.zeros_(self.w_2.bias)

    def forward(self, x):

        residual = x

        x = self.w_2(self.relu(self.w_1(x)))
        #x = self.dropout(x)
        x = self.addlayer(x, residual)

        x = self.layer_norm(x)

        return x
#____________ For Unboxed_________________________ 
class OneHead(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k, bias=False)
        self.w_ks = nn.Linear(d_model, d_k, bias=False)
        self.w_vs = nn.Linear(d_model, d_v, bias=False)

        self.attention = ScaledDotProductAttention_unboxed(temperature=d_k ** 0.5)

    def forward(self, q, k, v, mask=None):
        # q, k, v are of size b x lq x d_model

        # Pass through the pre-attention projection: ouput= b x lq x (dv or dk)..but d_v=d_k=d_model/h 
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        #if mask is not None:
        #    mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q = self.attention(q, k, v, mask=mask)
        # ouput_size = b x lq x dv
        return q

class MultiHeadAttention_unboxed(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.attention_stack = nn.ModuleList([
            OneHead(d_model, d_k, d_v, dropout=dropout)
            for _ in range(n_head)])
        
        self.concat = _concatenateLayer()
        self.contiguous =_contiguous()
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        #self.dropout = nn.Dropout(dropout)
        self.addlayer = _addLayer()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None, return_attns=False):
        head_output_list=[]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        #sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        
        for head_layer in self.attention_stack:
            head_output = head_layer(q, k, v, mask)
            head_output_list += [head_output] 
        # Concatenate all the heads together: b x lq x (n*dv)
        q = self.concat(2, *head_output_list) # concat along dim=2, output_size =b * lq *(n*dv) 
        q = self.contiguous(q)

        q = self.fc(q)#q = self.dropout(self.fc(q))
        q = self.addlayer(q,residual)

        q = self.layer_norm(q)

        return q
##########################################################

# Layers

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, unboxed=True):
        super(EncoderLayer, self).__init__()
        if  unboxed:
            self.slf_attn = MultiHeadAttention_unboxed(n_head, d_model, d_k, d_v, dropout=dropout)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, unboxed=True):
        super(DecoderLayer, self).__init__()
        if  unboxed:
            self.slf_attn = MultiHeadAttention_unboxed(n_head, d_model, d_k, d_v, dropout=dropout)
            self.enc_attn = MultiHeadAttention_unboxed(n_head, d_model, d_k, d_v, dropout=dropout)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output = self.slf_attn(
            dec_input, dec_input, dec_input, slf_attn_mask)
        dec_output = self.enc_attn(
            dec_output, enc_output, enc_output, dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output

#############################################################
# Model


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200, dropout=0.1): # d_hid is word_vec_len (=d_model = 512), n_position is sentence seq lenght 
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.addlayer = _addLayer()
        #self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_hid, eps=1e-6)
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # x.size() = batch_size, seq_len, word_vec_len (d_model)
        # self.pos_table[:, :x.size(1)].size() = 1, seq_len, word_vec_len
        #print("size a:", x.size())
        #print("Size b:", self.pos_table[:, :x.size(1)].size())
        encoded = self.addlayer(x, self.pos_table[:, :x.size(1)].clone().detach())
        encoded = self.layer_norm(encoded)#encoded = self.layer_norm(self.dropout(encoded))
        return encoded


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False, unboxed=True):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position, dropout=dropout)
        
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, unboxed=unboxed)
            for _ in range(n_layers)])
        
        self.scale_emb = scale_emb
        self.d_model = d_model

        const_w = 1/(10**3)
        torch.nn.init.constant_(self.src_word_emb.weight, const_w )

    def forward(self, src_seq, src_mask, return_attns=False):

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.position_enc(enc_output)

        for enc_layer in self.layer_stack:
            enc_output= enc_layer(enc_output, slf_attn_mask=src_mask)

        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False, unboxed=True):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position, dropout=dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, unboxed=unboxed)
            for _ in range(n_layers)])

        self.scale_emb = scale_emb
        self.d_model = d_model

        const_w = 1/(10**3)
        torch.nn.init.constant_(self.trg_word_emb.weight, const_w )

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.position_enc(dec_output)

        for dec_layer in self.layer_stack:
            dec_output= dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)

        return dec_output

if with_gpu_split:  
    class Transformer(nn.Module):
        ''' A sequence to sequence model with attention mechanism. '''

        def __init__(
                self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
                d_word_vec=512, d_model=512, d_inner=2048,
                n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
                trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
                scale_emb_or_prj='prj', unboxed=True):

            super().__init__()

            self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

            # In section 3.4 of paper "Attention Is All You Need", there is such detail:
            # "In our model, we share the same weight matrix between the two
            # embedding layers and the pre-softmax linear transformation...
            # In the embedding layers, we multiply those weights by \sqrt{d_model}".
            #
            # Options here:
            #   'emb': multiply \sqrt{d_model} to embedding output
            #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
            #   'none': no multiplication

            assert scale_emb_or_prj in ['emb', 'prj', 'none']
            scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
            #self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
            self.d_model = d_model
            
            self.encoder = Encoder(
                n_src_vocab=n_src_vocab, n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb, unboxed=unboxed).to(0)

            self.decoder = Decoder(
                n_trg_vocab=n_trg_vocab, n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb, unboxed=unboxed).to(1)

            self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False).to(1)
            #if self.scale_prj:
            #    self.multiply_scale = _multiply()

            #for p in self.parameters():
            #    if p.dim() > 1:
            #       nn.init.xavier_uniform_(p) 

            assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
            the dimensions of all module outputs shall be the same.'

            if trg_emb_prj_weight_sharing:
                # Share the weight between target word embedding & last dense layer
                self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

            if emb_src_trg_weight_sharing:
                self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


        def forward(self, src_seq, trg_seq):
            src_seq = src_seq.to(0)
            trg_seq = trg_seq.to(1)
            
            src_mask = get_pad_mask(src_seq, self.src_pad_idx)
            src_mask0 = src_mask.to(0)
            src_mask1 = src_mask.to(1)
            trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq).to(1)
            
            enc_output = self.encoder(src_seq, src_mask0)
            enc_output = enc_output.to(1)
            dec_output = self.decoder(trg_seq, trg_mask, enc_output, src_mask1)
            seq_logit = self.trg_word_prj(dec_output)
            #if self.scale_prj:
            #    seq_logit = self.multiply_scale(seq_logit, (self.d_model ** -0.5))
                #seq_logit *= self.d_model ** -0.5
                
            #print("size of output:", seq_logit.size()) 
            #return seq_logit.view(-1, seq_logit.size(2)) # why this? 
            return seq_logit

else:
    class Transformer(nn.Module):
        ''' A sequence to sequence model with attention mechanism. '''

        def __init__(
                self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
                d_word_vec=512, d_model=512, d_inner=2048,
                n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
                trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
                scale_emb_or_prj='prj', unboxed=True):

            super().__init__()

            self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

            # In section 3.4 of paper "Attention Is All You Need", there is such detail:
            # "In our model, we share the same weight matrix between the two
            # embedding layers and the pre-softmax linear transformation...
            # In the embedding layers, we multiply those weights by \sqrt{d_model}".
            #
            # Options here:
            #   'emb': multiply \sqrt{d_model} to embedding output
            #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
            #   'none': no multiplication

            assert scale_emb_or_prj in ['emb', 'prj', 'none']
            scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
            #self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
            self.d_model = d_model
            
            self.encoder = Encoder(
                n_src_vocab=n_src_vocab, n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb, unboxed=unboxed)

            self.decoder = Decoder(
                n_trg_vocab=n_trg_vocab, n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb, unboxed=unboxed)

            self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

            const_w = 1/(10**3)
            torch.nn.init.constant_(self.trg_word_prj.weight, const_w )

            #if self.scale_prj:
            #    self.multiply_scale = _multiply()

            #for p in self.parameters():
            #    if p.dim() > 1:
            #        nn.init.xavier_uniform_(p) 

            assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
            the dimensions of all module outputs shall be the same.'

            if trg_emb_prj_weight_sharing:
                # Share the weight between target word embedding & last dense layer
                self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

            if emb_src_trg_weight_sharing:
                self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


        def forward(self, src_seq, trg_seq):
            print("This is the repetable Transformer")
            src_mask = get_pad_mask(src_seq, self.src_pad_idx)
            trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
            
            enc_output = self.encoder(src_seq, src_mask)
            dec_output = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
            seq_logit = self.trg_word_prj(dec_output)
            #if self.scale_prj:
            #    seq_logit = self.multiply_scale(seq_logit, (self.d_model ** -0.5))
                #seq_logit *= self.d_model ** -0.5
                
            #print("size of output:", seq_logit.size()) 
            #return seq_logit.view(-1, seq_logit.size(2)) # why this? 
            return seq_logit
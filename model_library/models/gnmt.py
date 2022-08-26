## Source: https://github.com/NVIDIA/DeepLearningExamples/tree/26d8955cc5ffe865bf83249c125e65f7f72781d8/PyTorch/Translation/GNMT
## Basic GNMT experiment in /home/cshetty2/sct/pytorch/basic_experiments/experiments.ipynb

import torch
from torch import optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
# use of pack_padded_sequence:
#https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch/55805785#55805785
import torch.nn.init as init

import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.nn.functional import log_softmax
import numpy as np

class _addLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1+x2

class _tanhLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        return torch.tanh(x1)

class _matmulLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1.matmul(x2)

class _concatenateLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *x):
        return torch.cat(x, 1)

class _concatenateTwo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=2)

class _maskedFill(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x , mask):
        #print(mask)
        return x.masked_fill_(mask,  -65504.0)

class _bmm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores_normalized , keys):
        return torch.bmm(scores_normalized, keys)

# end /defined by chirag/

###################### Hyper parameter ##########################################
## same as baechi tensorflow
vocab_size = 30000  
min_sequence_length = 5
max_sequence_length = 40 
#embedding_len = 1024
lstm_hidden_size = 512
lstm_no_layers = 4# Actual: 4 ## applies to both encoder and decoder

##################### Embedding ###################################################
## nn.Embedding (Vocabulary_length,vector_length)
# Each word in the vocab is represented as a vector of lenght vector_length. Initilization is random

#embedding = nn.Embedding(vocab_size,embedding_len)
#embedding(torch.LongTensor([[999,999,4],[10,2,3]]))

######################## ENCODER ####################################################

class Config():
    def __init__(self):
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.BOS_TOKEN = '<s>'
        self.EOS_TOKEN = '<\s>'

        # special PAD, UNKNOWN, BEGIN-OF-STRING, END-OF-STRING tokens
        self.PAD, self.UNK, self.BOS, self.EOS = [0, 1, 2, 3]

        # path to the moses detokenizer, relative to the data directory
        self.DETOKENIZER = 'mosesdecoder/scripts/tokenizer/detokenizer.perl'

config = Config()
##-----------------------------------------------------------------------------
def init_lstm_(lstm, init_weight=0.1):
    """
    Initializes weights of LSTM layer.
    Weights and biases are initialized with uniform(-init_weight, init_weight)
    distribution.
    :param lstm: instance of torch.nn.LSTM
    :param init_weight: range for the uniform initializer
    """
    # Initialize hidden-hidden weights
    init.uniform_(lstm.weight_hh_l0.data, -init_weight, init_weight)
    # Initialize input-hidden weights:
    init.uniform_(lstm.weight_ih_l0.data, -init_weight, init_weight)

    # Initialize bias. PyTorch LSTM has two biases, one for input-hidden GEMM
    # and the other for hidden-hidden GEMM. Here input-hidden bias is
    # initialized with uniform distribution and hidden-hidden bias is
    # initialized with zeros.
    init.uniform_(lstm.bias_ih_l0.data, -init_weight, init_weight)
    init.zeros_(lstm.bias_hh_l0.data)

    if lstm.bidirectional:
        init.uniform_(lstm.weight_hh_l0_reverse.data, -init_weight, init_weight)
        init.uniform_(lstm.weight_ih_l0_reverse.data, -init_weight, init_weight)

        init.uniform_(lstm.bias_ih_l0_reverse.data, -init_weight, init_weight)
        init.zeros_(lstm.bias_hh_l0_reverse.data)

#----------------------------------------------------------------------------
class ResidualRecurrentEncoder(nn.Module):
    """
    Encoder with Embedding, LSTM layers, residual connections and optional
    dropout.
    The first LSTM layer is bidirectional and uses variable sequence length
    API, the remaining (num_layers-1) layers are unidirectional. Residual
    connections are enabled after third LSTM layer, dropout is applied on
    inputs to LSTM layers.
    """
    def __init__(self, vocab_size, hidden_size=lstm_hidden_size, num_layers=lstm_no_layers, dropout=0.2,
                 batch_first=False, embedder=None, init_weight=0.1):
        """
        Constructor for the ResidualRecurrentEncoder.
        :param vocab_size: size of vocabulary
        :param hidden_size: hidden size for LSTM layers
        :param num_layers: number of LSTM layers, 1st layer is bidirectional
        :param dropout: probability of dropout (on input to LSTM layers)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param embedder: instance of nn.Embedding, if None constructor will
            create new embedding layer
        :param init_weight: range for the uniform initializer
        """
        super(ResidualRecurrentEncoder, self).__init__()
        self.batch_first = batch_first
        self.rnn_layers = nn.ModuleList()
        # 1st LSTM layer, bidirectional
        self.rnn_layers.append(
            nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True,
                    batch_first=batch_first, bidirectional=True))

        # 2nd LSTM layer, with 2x larger input_size
        self.rnn_layers.append(
            nn.LSTM((2 * hidden_size), hidden_size, num_layers=1, bias=True,
                    batch_first=batch_first))

        # Remaining LSTM layers
        for _ in range(num_layers - 2):
            self.rnn_layers.append(
                nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True,
                        batch_first=batch_first))

        for lstm in self.rnn_layers:
            init_lstm_(lstm, init_weight)

        self.dropout = nn.Dropout(p=dropout)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size,
                                         padding_idx=config.PAD)
            nn.init.uniform_(self.embedder.weight.data, -init_weight,
                             init_weight)
        self.addLayers = nn.ModuleList()
        for i in range(2,len(self.rnn_layers)):
            self.addLayers.append(_addLayer())

    def forward(self, inputs, lengths):
        """
        Execute the encoder.
        :param inputs: tensor with indices from the vocabulary
        :param lengths: vector with sequence lengths (excluding padding)
        returns: tensor with encoded sequences
        """
        x = self.embedder(inputs)

        # bidirectional layer
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths.cpu().numpy(),
                                 batch_first=self.batch_first)
        x, _ = self.rnn_layers[0](x)
        x, _ = pad_packed_sequence(x, batch_first=self.batch_first)

        # 1st unidirectional layer
        x = self.dropout(x)
        x, _ = self.rnn_layers[1](x)

        # the rest of unidirectional layers,
        # with residual connections starting from 3rd layer
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = self.dropout(x)
            x, _ = self.rnn_layers[i](x)
            x = self.addLayers[i-2](x,residual)

        return x

# ##### Check #############
# vocab_len = 50000
# model = ResidualRecurrentEncoder(vocab_len, batch_first=True)
# ## Forward inputs
# ## 1. Sentences as sequences of words (indices in 0 to vocab_length)
# ## 2. Lengths of the sentences (Since we can't have a 2d torch tensor where each row is different length, the torch tensor
# ## will have size of the longest sentences and we specify length of each sentences seperately.)
# model.forward(torch.LongTensor([[50000-1,4,5],[50000-1,4,0]]), torch.tensor([3,2]))
# #######################

################################ Attention ################################################

class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (https://arxiv.org/abs/1409.0473)
    Implementation is very similar to tf.contrib.seq2seq.BahdanauAttention
    """
    def __init__(self, query_size, key_size, num_units, normalize=False,
                 batch_first=False, init_weight=0.1):
        """
        Constructor for the BahdanauAttention.
        :param query_size: feature dimension for query
        :param key_size: feature dimension for keys
        :param num_units: internal feature dimension
        :param normalize: whether to normalize energy term
        :param batch_first: if True batch size is the 1st dimension, if False
            the sequence is first and batch size is second
        :param init_weight: range for uniform initializer used to initialize
            Linear key and query transform layers and linear_att vector
        """
        super(BahdanauAttention, self).__init__()

        self.normalize = normalize
        self.batch_first = batch_first
        self.num_units = num_units

        self.linear_q = nn.Linear(query_size, num_units, bias=False)
        self.linear_k = nn.Linear(key_size, num_units, bias=False)
        nn.init.uniform_(self.linear_q.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.linear_k.weight.data, -init_weight, init_weight)

        self.linear_att = Parameter(torch.Tensor(num_units))

        self.mask = None
        self.maskFill  = _maskedFill()
        self.bmm = _bmm()

        if self.normalize:
            self.normalize_scalar = Parameter(torch.Tensor(1))
            self.normalize_bias = Parameter(torch.Tensor(num_units))
        else:
            self.register_parameter('normalize_scalar', None)
            self.register_parameter('normalize_bias', None)

        self.reset_parameters(init_weight)

        self.addLayer1 = _addLayer()
        self.addLayer2 = _addLayer()
        self.tanhLayer = _tanhLayer()
        self.matmulLayer = _matmulLayer()

    def reset_parameters(self, init_weight):
        """
        Sets initial random values for trainable parameters.
        """
        stdv = 1. / np.sqrt(self.num_units)
        self.linear_att.data.uniform_(-init_weight, init_weight)

        if self.normalize:
            self.normalize_scalar.data.fill_(stdv)
            self.normalize_bias.data.zero_()

    def set_mask(self, context_len, context):
        """
        sets self.mask which is applied before softmax
        ones for inactive context fields, zeros for active context fields
        :param context_len: b
        :param context: if batch_first: (b x t_k x n) else: (t_k x b x n)
        self.mask: (b x t_k)
        """

        if self.batch_first:
            max_len = context.size(1)
        else:
            max_len = context.size(0)

        indices = torch.arange(0, max_len, dtype=torch.int64,
                               device=context.device)
        self.mask = indices >= (context_len.unsqueeze(1))

    def calc_score(self, att_query, att_keys):
        """
        Calculate Bahdanau score
        :param att_query: b x t_q x n
        :param att_keys: b x t_k x n
        returns: b x t_q x t_k scores
        """

        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)

        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = self.addLayer1(att_query, att_keys)
        #sum_qk = att_query + att_keys

        if self.normalize:
            sum_qk = self.addLayer2(sum_qk, self.normalize_bias)
            #sum_qk = sum_qk + self.normalize_bias
            linear_att = self.linear_att / self.linear_att.norm()
            linear_att = linear_att * self.normalize_scalar
        else:
            linear_att = self.linear_att

        out = self.matmulLayer(self.tanhLayer(sum_qk), linear_att) 
        #out = torch.tanh(sum_qk).matmul(linear_att)
        return out

    def forward(self, query, keys):
        """
        :param query: if batch_first: (b x t_q x n) else: (t_q x b x n)
        :param keys: if batch_first: (b x t_k x n) else (t_k x b x n)
        :returns: (context, scores_normalized)
        context: if batch_first: (b x t_q x n) else (t_q x b x n)
        scores_normalized: if batch_first (b x t_q x t_k) else (t_q x b x t_k)
        """

        # first dim of keys and query has to be 'batch', it's needed for bmm
        if not self.batch_first:
            keys = keys.transpose(0, 1)
            if query.dim() == 3:
                query = query.transpose(0, 1)

        if query.dim() == 2:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False

        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)

        # FC layers to transform query and key
        processed_query = self.linear_q(query)
        processed_key = self.linear_k(keys)

        # scores: (b x t_q x t_k)
        scores = self.calc_score(processed_query, processed_key)

        if self.mask is not None:
            mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
            # I can't use -INF because of overflow check in pytorch
            #print(scores)
            #print(mask)
            self.maskFill(scores, mask)
            #scores.masked_fill_(mask, -65504.0)

        # Normalize the scores, softmax over t_k
        scores_normalized = F.softmax(scores, dim=-1)

        # Calculate the weighted average of the attention inputs according to
        # the scores
        # context: (b x t_q x n)
        context = self.bmm(scores_normalized, keys)

        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        elif not self.batch_first:
            context = context.transpose(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)

        return context, scores_normalized

# ######## Check ##############
# model = BahdanauAttention(1024,1024,1024)
# #############################


################################ Decoder ##########################################
class RecurrentAttention(nn.Module):
    """
    LSTM wrapped with an attention module.
    """
    def __init__(self, input_size=1024, context_size=1024, hidden_size=lstm_hidden_size,
                 num_layers=1, batch_first=False, dropout=0.2,
                 init_weight=0.1):
        """
        Constructor for the RecurrentAttention.
        :param input_size: number of features in input tensor
        :param context_size: number of features in output from encoder
        :param hidden_size: internal hidden size
        :param num_layers: number of layers in LSTM
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param dropout: probability of dropout (on input to LSTM layer)
        :param init_weight: range for the uniform initializer
        """

        super(RecurrentAttention, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bias=True,
                           batch_first=batch_first)
        init_lstm_(self.rnn, init_weight)

        self.attn = BahdanauAttention(hidden_size, context_size, context_size,
                                      normalize=True, batch_first=batch_first)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, context, context_len):
        """
        Execute RecurrentAttention.
        :param inputs: tensor with inputs
        :param hidden: hidden state for LSTM layer
        :param context: context tensor from encoder
        :param context_len: vector of encoder sequence lengths
        :returns (rnn_outputs, hidden, attn_output, attn_scores)
        """
        # set attention mask, sequences have different lengths, this mask
        # allows to include only valid elements of context in attention's
        # softmax
        self.attn.set_mask(context_len, context)

        inputs = self.dropout(inputs)
        rnn_outputs, hidden = self.rnn(inputs, hidden)
        attn_outputs, scores = self.attn(rnn_outputs, context)

        return rnn_outputs, hidden, attn_outputs, scores


class Classifier(nn.Module):
    """
    Fully-connected classifier
    """
    def __init__(self, in_features, out_features, init_weight=0.1):
        """
        Constructor for the Classifier.
        :param in_features: number of input features
        :param out_features: number of output features (size of vocabulary)
        :param init_weight: range for the uniform initializer
        """
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_features, out_features)
        nn.init.uniform_(self.classifier.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.classifier.bias.data, -init_weight, init_weight)

    def forward(self, x):
        """
        Execute the classifier.
        :param x: output from decoder
        """
        out = self.classifier(x)
        return out


class ResidualRecurrentDecoder(nn.Module):
    """
    Decoder with Embedding, LSTM layers, attention, residual connections and
    optinal dropout.
    Attention implemented in this module is different than the attention
    discussed in the GNMT arxiv paper. In this model the output from the first
    LSTM layer of the decoder goes into the attention module, then the
    re-weighted context is concatenated with inputs to all subsequent LSTM
    layers in the decoder at the current timestep.
    Residual connections are enabled after 3rd LSTM layer, dropout is applied
    on inputs to LSTM layers.
    """
    def __init__(self, vocab_size, hidden_size=lstm_hidden_size, num_layers=lstm_no_layers, dropout=0.2,
                 batch_first=False, embedder=None, init_weight=0.1):
        """
        Constructor of the ResidualRecurrentDecoder.
        :param vocab_size: size of vocabulary
        :param hidden_size: hidden size for LSMT layers
        :param num_layers: number of LSTM layers
        :param dropout: probability of dropout (on input to LSTM layers)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param embedder: instance of nn.Embedding, if None constructor will
            create new embedding layer
        :param init_weight: range for the uniform initializer
        """
        super(ResidualRecurrentDecoder, self).__init__()

        self.num_layers = num_layers

        self.att_rnn = RecurrentAttention(hidden_size, hidden_size,
                                          hidden_size, num_layers=1,
                                          batch_first=batch_first,
                                          dropout=dropout)

        self.rnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.rnn_layers.append(
                nn.LSTM(2 * hidden_size, hidden_size, num_layers=1, bias=True,
                        batch_first=batch_first))

        for lstm in self.rnn_layers:
            init_lstm_(lstm, init_weight)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size,
                                         padding_idx=config.PAD)
            nn.init.uniform_(self.embedder.weight.data, -init_weight,
                             init_weight)

        self.classifier = Classifier(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

        self.addLayers = nn.ModuleList()
        for i in range(1, len(self.rnn_layers)):
            self.addLayers.append(_addLayer())

        #self.concatenateLayer1 = _concatenateLayer() # Only requirec for inference
        self.concatenateLayer2 = _concatenateTwo()

        self.concatenateLayers = nn.ModuleList()
        for i in range(1, len(self.rnn_layers)):
            self.concatenateLayers.append(_concatenateTwo())


    def init_hidden(self, hidden):
        """
        Converts flattened hidden state (from sequence generator) into a tuple
        of hidden states.
        :param hidden: None or flattened hidden state for decoder RNN layers
        """
        if hidden is not None:
            # per-layer chunks
            hidden = hidden.chunk(self.num_layers)
            # (h, c) chunks for LSTM layer
            hidden = tuple(i.chunk(2) for i in hidden)
        else:
            hidden = [None] * self.num_layers

        self.next_hidden = []
        return hidden

    def append_hidden(self, h):
        """
        Appends the hidden vector h to the list of internal hidden states.
        :param h: hidden vector
        """
        if self.inference:
            self.next_hidden.append(h)

    def package_hidden(self):
        """
        Flattens the hidden state from all LSTM layers into one tensor (for
        the sequence generator).
        """
        if self.inference:
            #hidden = self.concatenateLayer1(tuple(itertools.chain(*self.next_hidden)))
            hidden = torch.cat(tuple(itertools.chain(*self.next_hidden)))
        else:
            hidden = None
        return hidden

    def forward(self, inputs, context, inference=False):
        """
        Execute the decoder.
        :param inputs: tensor with inputs to the decoder
        :param context: state of encoder, encoder sequence lengths and hidden
            state of decoder's LSTM layers
        :param inference: if True stores and repackages hidden state
        """
        self.inference = inference

        enc_context, enc_len, hidden = context
        hidden = self.init_hidden(hidden)

        x = self.embedder(inputs)

        x, h, attn, scores = self.att_rnn(x, hidden[0], enc_context, enc_len)
        self.append_hidden(h)

        x = self.concatenateLayer2(x,attn)
        #x = torch.cat((x, attn), dim=2)
        x = self.dropout(x)
        x, h = self.rnn_layers[0](x, hidden[1])
        self.append_hidden(h)

        for i in range(1, len(self.rnn_layers)):
            residual = x
            x = self.concatenateLayers[i-1](x, attn)
            #x = torch.cat((x, attn), dim=2)
            x = self.dropout(x)
            x, h = self.rnn_layers[i](x, hidden[i + 1])
            self.append_hidden(h)
            x = self.addLayers[i-1](x, residual)
            #x = x + residual

        x = self.classifier(x)
        hidden = self.package_hidden()

        return x, scores, [enc_context, enc_len, hidden]

# ########### Check ##########
# model = ResidualRecurrentDecoder(50000)
# ###########################

########################## Seq2seq Base ############################################

class Seq2Seq(nn.Module):
    """
    Generic Seq2Seq module, with an encoder and a decoder.
    """
    def __init__(self, encoder=None, decoder=None, batch_first=False):
        """
        Constructor for the Seq2Seq module.
        :param encoder: encoder module
        :param decoder: decoder module
        :param batch_first: if True the model uses (batch, seq, feature)
            tensors, if false the model uses (seq, batch, feature) tensors
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_first = batch_first

    def encode(self, inputs, lengths):
        """
        Applies the encoder to inputs with a given input sequence lengths.
        :param inputs: tensor with inputs (batch, seq_len) if 'batch_first'
            else (seq_len, batch)
        :param lengths: vector with sequence lengths (excluding padding)
        """
        return self.encoder(inputs, lengths)

    def decode(self, inputs, context, inference=False):
        """
        Applies the decoder to inputs, given the context from the encoder.
        :param inputs: tensor with inputs (batch, seq_len) if 'batch_first'
            else (seq_len, batch)
        :param context: context from the encoder
        :param inference: if True inference mode, if False training mode
        """
        return self.decoder(inputs, context, inference)

    def generate(self, inputs, context, beam_size):
        """
        Autoregressive generator, works with SequenceGenerator class.
        Executes decoder (in inference mode), applies log_softmax and topK for
        inference with beam search decoding.
        :param inputs: tensor with inputs to the decoder
        :param context: context from the encoder
        :param beam_size: beam size for the generator
        returns: (words, logprobs, scores, new_context)
            words: indices of topK tokens
            logprobs: log probabilities of topK tokens
            scores: scores from the attention module (for coverage penalty)
            new_context: new decoder context, includes new hidden states for
                decoder RNN cells
        """
        logits, scores, new_context = self.decode(inputs, context, True)
        logprobs = log_softmax(logits, dim=-1)
        logprobs, words = logprobs.topk(beam_size, dim=-1)
        return words, logprobs, scores, new_context

############################ GNMT ################################################

class GNMT(Seq2Seq):
    """
    GNMT v2 model
    """
    def __init__(self, vocab_size, hidden_size=lstm_hidden_size, num_layers=lstm_no_layers, dropout=0.2,
                 batch_first=False, share_embedding=True):
        """
        Constructor for the GNMT v2 model.
        :param vocab_size: size of vocabulary (number of tokens)
        :param hidden_size: internal hidden size of the model
        :param num_layers: number of layers, applies to both encoder and
            decoder
        :param dropout: probability of dropout (in encoder and decoder)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param share_embedding: if True embeddings are shared between encoder
            and decoder
        """

        super(GNMT, self).__init__(batch_first=batch_first)

        if share_embedding:
            embedder = nn.Embedding(vocab_size, hidden_size,
                                    padding_idx=config.PAD)
            nn.init.uniform_(embedder.weight.data, -0.1, 0.1)
        else:
            embedder = None

        self.encoder = ResidualRecurrentEncoder(vocab_size, hidden_size,
                                                num_layers, dropout,
                                                batch_first, embedder)

        self.decoder = ResidualRecurrentDecoder(vocab_size, hidden_size,
                                                num_layers, dropout,
                                                batch_first, embedder)

    def forward(self, input_encoder, input_enc_len, input_decoder):
        context = self.encode(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decode(input_decoder, context)

        return output


############### Util ##############################
## to print model
def recur_function_print(module, n):
    n = n+1
    sub_modules = module.__dict__['_modules']
    for name, sub_module in sub_modules.items():
        print("--"*(n-1), name)
        # sub modules of sub_module, if there are more than 1, we need further recursion
        sub_sub_modules = sub_module.__dict__['_modules']
        if len(sub_sub_modules) > 0:
            recur_function(sub_module, n)
            continue
# ####### check ##############
# recur_function(model, 0)
# ############################


# ######## Train template ##################

# vocab_size = 50
# model = GNMT(vocab_size, batch_first=True)

# padding_idx = 0
# criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')


# # try adam
# optimizer = optim.SGD(model.parameters(), lr = 0.0001); 

# ### encoder input => 2 * 3 (say E1*E2) - each element must be in [0, vocab_size-1]
# ### encoder lengths => 2 (E1) - each element must be <= E2
# ### Decoder inputs => 2 * 4 (E1*D1)
# out_forward = model.forward(torch.LongTensor([[2,4,3],[3,2,0]]), torch.tensor([3,1]), torch.LongTensor([[1,4,2,1],[4,2,0,2]]))
# out_size = out_forward.size()# [no_input_vectors, length_of_each_output_vector, vocab_size]
# labels = torch.empty(out_size[0],out_size[2], dtype=torch.long).random_(2) ## generates rand int in [0 ,2-1]

# loss = criterion(out_forward, labels)
# loss.backward(loss)
# optimizer.step()

# ###############################################
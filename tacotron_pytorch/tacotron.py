# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

from .attention import BahdanauAttention, AttentionWrapper
from .attention import get_mask_from_lengths
import sys
import math


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128]):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        # Following tensorflow's default parameters
        self.bn = nn.BatchNorm1d(out_dim, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-dim
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs


class Encoder(nn.Module):
    def __init__(self, in_dim):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_dim, sizes=[256, 128])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])

    def forward(self, inputs, input_lengths=None):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs, input_lengths)


class Decoder(nn.Module):
    def __init__(self, in_dim):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, 256)
        # (prenet_out + attention context) -> output
        self.memory_layer = nn.Linear(256, 256, bias=False)
        self.project_to_decoder_in = nn.Linear(512, 256)

        self.Attention = nn.ModuleList(
            [Attention(256, 256) for i in range(1)]
        )

        self.decoder_cnns = nn.ModuleList(
            [CNN_layer(256) for i in range(2)]
        )

        self.proj_to_mel = nn.Linear(256, in_dim)
        self.max_decoder_steps = 200

    def forward(self, encoder_outputs, first_embedding, inputs=None, memory_lengths=None, parallel=False):
        """
        Decoder forward step.

        If decoder inputs are not given (e.g., at testing time), as noted in
        Tacotron paper, greedy decoding is adapted.

        Args:
            encoder_outputs: Encoder outputs. (B, T_encoder, dim)
            inputs: Decoder inputs. i.e., mel-spectrogram. If None (at eval-time),
              decoder outputs are used as decoder inputs.
            memory_lengths: Encoder output (memory) lengths. If not None, used for
              attention masking.
        """
        B = encoder_outputs.size(0)

        processed_memory = self.memory_layer(encoder_outputs)
        if memory_lengths is not None:
            mask = get_mask_from_lengths(processed_memory, memory_lengths)
        else:
            mask = None

        if inputs is not None:
            T_decoder = inputs.size(1)

        # go frames
        initial_input = Variable(torch.zeros(B, 1, 256)).cuda()
        t = 0
        current_input = initial_input


        if parallel:
            inputs = self.linear(inputs)
            layer_output = inputs = torch.cat((initial_input, inputs), dim=1)
            inputs = inputs[:, :-1, :]
            layer_output = layer_output[:, :-1, :]
            for idx in range(len(self.decoder_cnns)):
                layer_output = self.decoder_cnns[idx](layer_output, parallel=parallel)
                if idx < len(self.decoder_cnns)-1:
                    output = self.Attention[idx](layer_output, encoder_outputs, inputs, first_embedding)
                    layer_output = layer_output + output

            outputs = layer_output
            outputs = self.proj_to_mel(outputs)


        else: 
            hidden_layers = [Variable(torch.zeros(B, 4, 256)).cuda() for i in range(4)]
            outputs = Variable(torch.zeros(B, 5, self.in_dim)).cuda()

            while True:
                layer_output = current_input = self.linear(outputs[:, -5:, :])
                for idx in range(len(self.decoder_cnns)):
                    hidden_output = self.decoder_cnns[idx](layer_output)
                    if idx < len(self.decoder_cnns)-1:
                        hidden_layers[idx] = torch.cat((hidden_layers[idx], hidden_output), dim = 1)
                        hidden_layers[idx] = layer_output = hidden_layers[idx][:, -5:, :]
                        output = self.Attention[idx](layer_output, encoder_outputs, current_input, first_embedding)
                        layer_output = layer_output + output
                    else:
                        final_output = self.proj_to_mel(hidden_output)
                        outputs = torch.cat((outputs, final_output), dim = 1)
                    
                t = t + 1

                if t > 1 and is_end_of_frames(output):
                    break
                elif t == 1000:
                    # print("Warning! doesn't seems to be converged")
                    break
            outputs = outputs[:, 4:, :]
        return outputs


def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()


class Tacotron(nn.Module):
    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False, parallel=True):
        super(Tacotron, self).__init__()
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.use_memory_mask = use_memory_mask
        self.embedding = nn.Embedding(n_vocab, embedding_dim,
                                      padding_idx=padding_idx)
        # Trying smaller std
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(mel_dim)

        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)
        self.parallel = parallel

    def forward(self, inputs, targets=None, input_lengths=None):
        B = inputs.size(0)

        inputs = self.embedding(inputs)
        # (B, T', in_dim)
        encoder_outputs = self.encoder(inputs, input_lengths)

        if self.use_memory_mask:
            memory_lengths = input_lengths
        else:
            memory_lengths = None
        # (B, T', mel_dim*r)
        mel_outputs = self.decoder(
            encoder_outputs, inputs, targets, memory_lengths=memory_lengths, parallel=self.parallel)

        # Post net processing below

        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs

class CNN_layer(nn.Module):
    def __init__(self, input_dim, k_size=5):
        super(CNN_layer, self).__init__()
        self.input_dim = input_dim
        self.k_size = k_size
        
        self.conv=nn.Conv1d(input_dim, input_dim*2, kernel_size=k_size)
    def forward(self, x, parallel=False):
        if parallel:
            padding = Variable(torch.zeros(x.size(0), self.k_size-1, x.size(2))).cuda()
            inputs = torch.cat((padding, x), dim = 1)
        else: 
            inputs = x 
        inputs = inputs.transpose(1, 2)
        y = self.conv(inputs)
        y = F.glu(y.transpose(1, 2))
        k = math.sqrt(2)
        if parallel: 
            return (y+x) / k
        else:
            
            return (y+x[:, -1:, :]) / k


class Attention(nn.Module):
    def __init__(self, input_dim, encode_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.encode_dim = encode_dim
        self.layer = nn.Linear(input_dim, input_dim)
    def forward(self, query, key, inputs, first_embedding):
        output = self.layer(query) #+ inputs
        attn = torch.bmm(output, key.transpose(1, 2))
        attn = F.softmax(attn, dim=2)
        c_vector = torch.bmm(attn, key+first_embedding)
        return c_vector


   

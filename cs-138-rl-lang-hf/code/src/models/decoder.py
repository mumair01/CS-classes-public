# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    """
    Implements different attention mechanisms to be used with the Decoder such
    that there is less information loss.
    """

    ATTN_METHODS = ['dot', 'general', 'concat']

    def __init__(self, attn_method, hidden_size):
        """
        Args:
            attn_method (str): Type of attention method - can be dot, general,
                    or concat.
            hidden_size (int): The number of features.
        """
        super().__init__()
        self.attn_method = attn_method
        if self.attn_method not in self.ATTN_METHODS:
            raise ValueError(self.method, 'unacceptable attention method')
        self.hidden_size = hidden_size
        if self.attn_method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.attn_method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        self.energy_methods = {
            'dot': self.dot_score,
            'general': self.general_score,
            'concat': self.concat_score
        }

    def dot_score(self, hidden, encoder_output):
        """
        Args:
            hidden (tensor)
            encoder_output (tensor)
        """
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        """
        Args:
            hidden (tensor)
            encoder_output (tensor)
        """
        energy = self.attn(encoder_output)
        return torch, sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        """
        Args:
            hidden (tensor)
            encoder_output (tensor)
        """
        energy = self.attn(torch.cat((hidden.expand(
            encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden (tensor)
            encoder_output (tensor)
        """
        # Calculate the attention weights / energies based on given method
        # and transpose the max_length and batch_size dimensions.
        attn_energies = self.energy_methods[self.attn_method](
            hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        # Return softmax normalized probability scores with added dimension.
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    """
    Decoder RNN based on the Luong Attention mechanism.

    Computation graph:
        1. Get embedding of the current word.
        2. Forward through unidirectional GRU.
        3. Calculate attention weights from the GRU output.
        4. Multiply attention weights to encoder output to get new context
            vector.
        5. Concatenate weighted context vector using Luong equation.
        6. Predict the next word.
        7. Return the output and final hidden state.
    """

    def __init__(self, attn_method, embedding, hidden_size, output_size,
                 n_layers=1, dropout=0.1):
        # -- Vars.
        super().__init__()
        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        # -- Layers defintiion.
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.attn = Attn(attn_method, hidden_size)

    def forward(self, input_word, last_hidden, encoder_outputs):
        # Embedding of current word
        embedded = self.embedding(input_word)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate the attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiple attention weights to encoder outputs for new context vector
        # NOTE: bmm is the batch matrix product - note attn_weights are batch 1
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        # Concatenate on dim 1
        concat_input = torch.cat([rnn_output, context], 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden

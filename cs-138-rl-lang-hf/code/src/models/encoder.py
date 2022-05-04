# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    """
    This is the encoder for a Seq2Seq model which encodes a variable
    length input sequence to a fixed-length context vector (the final hidden
    layer of the RNN).

    Computation graph:
        1. Convert word indices to embeddings.
        2. Pack padded batch of sequences.
        3. Forward pass through GRU.
        4. Unpack padding.
        5. Sum bidirectional GRU outputs.
        6. Return outputs and final hidden states.
    """

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        """
        Args:
            hidden_size (int): The number of features for the GRU.
                    Equal to the input size for GRU.
            embedding (nn.Embedding): Simple lookup table to store embeddings
                    of a fixed dictionary and size.
            n_layers (int): Number of layers of the GRU.
            dropout (float): Probability for the dropout layer.
                    If 0, no dropout layer is introduced.
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        # Initialize the GRU
        self.gru = nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=n_layers, dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        Args:
            input_seq (tensor): The minibatch
            input_lengths (tensor): Length of each input sequence.
            hidden (tensor): Previous hidden state

        Returns:
            (tensor): The output of the encoder with shape (L,N, D * H_out)
            (tensor): Hidden state of the encoder with shape
                    (2 * num_layers, N, H_out)

        """
        # Convert word indices to embeddings
        # output shape: (input_shape, embedding_dim) = (*, H)
        embedded = self.embedding(input_seq)
        # Pack the padded batch of sequences for the RNN model.
        # Performed primarily to save computation
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through the GRU
        # Input-shape: (L,N,H_in) , (2 * num_layers, N, H_out)
        # Output-shape: (L,N, D * H_out), (2 * num_layers, N, H_out)
        outputs, hidden = self.gru(packed, hidden)
        # Unpack the padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum the bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        # Return the outputs and the hidden state
        return outputs, hidden

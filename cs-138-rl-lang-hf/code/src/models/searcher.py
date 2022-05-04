# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..vars import *


class GreedySearchDecoder(nn.Module):
    """
    Decoding method when training is not using teacher forcing.
    At each timestep, the decoder output with the highest softmax value is chosen.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, input_length, max_length):
        # Foreard through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoders final hidden state to be decoder's first hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS token.
        # TODO: SOS token has to be defined here.
        decoder_input = torch.ones(
            1, 1, device=self.device, dtype=torch.long) * SOS_TOKEN
        # Initialize tensores to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and use its softmax
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record tokens and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Use current token as the next decoder input
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores

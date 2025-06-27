import torch
import torch.nn.functional as F
import torch.nn as nn
import math 
import numpy as np
from encoder import TransformerEncoder
from decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

    def forward(self, imgs, tokens):
        """
        Args:
            imgs: (B, C, H, W) input images.
                - B = batch size (number of images)
                - C = number of channels (e.g. 1 for grayscale, 3 for RGB)
                - H = height of the image
                - W = width of the image
            tokens: (B, T) integer IDs of input tokens for the decoder.
                - B = batch size (number of images)
                - T = max seq len, so max num of tokens in a sequence.
                - Each entry is an index, e.g: <s>, 3, 7, </s>, <pad>, <pad>, ...
        Returns:
            logits: (B, T, V) output logits for each token in the sequence.
                - B = batch size (number of images)
                - T = max seq len, so max num of tokens in a sequence.
                - V = size of the vocabulary (number of unique tokens)
                - Each entry is a logit for the corresponding token in the sequence.
                - The logits can be used to compute the loss and accuracy.
        """

        enc_out = self.encoder(imgs)
        logits = self.decoder(tokens, enc_out)
        return logits
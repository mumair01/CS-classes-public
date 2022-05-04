# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2021-11-29 18:23:29
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2021-11-30 09:30:45

from typing import List, Tuple, Dict
import unicodedata
import re
import itertools
import torch
import torch.nn as nn
import numpy as np
import os
from .voc import Voc
from .vars import *


# --- GENERAL UTILITY METHODS

def printLines(file_path: str, n: int = 10) -> None:
    """
    Print the first n lines of the given file.
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[:n]:
        print(line)


def unicodeToAscii(s: str) -> str:
    """
    Convert unicode symbols in the string to ascii
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s: str) -> str:
    """
    Lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def zeroPadding(l: List, fillvalue=PAD_TOKEN):
    """
    Pad the sentence based on the longest value.
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l: List[str]) -> List[bool]:
    """
    Create a binary mask from the sentences.
    Mask is True for any non PAD_TOKEN word and False otherwise.
    """
    m = list()
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            m[i].append(0) if token == PAD_TOKEN else m[i].append(1)
    return m


# ---- PAIRS METHODS


def loadLines(file_path: str, delimiter: str = '__eou__') -> List[List[str]]:
    """
    Split each line of the given file into a dictionary of fields.
    Specific for dialydialog dataset.
    """
    conversations = list()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split(delimiter)
            conversations.append(values)
    return conversations


def extractSentencePairs(conversations: List[List[str]]) -> List:
    """
    Extract pairs from the conversations in the data file.

    Returns a list of pairs.
    """
    qa_pairs = list()
    for conversation in conversations:
        # Iterate over all lines in the conversation,
        # ignoring the first line.
        for i in range(len(conversation) - 1):
            inputLine = conversation[i].strip()
            targetLine = conversation[i+1].strip()
            # Filter wrong samples
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def isPairLengthValid(pair: List[str], max_length: int):
    """
    Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH
    threshold
    """
    # Input sequences need to preserve the last word for EOS token
    return len(pair[0].split(' ')) < max_length and len(pair[1].split(' ')) < \
        max_length


def filterPairs(pairs: List, max_length: int):
    """
    Returns pairs such that both the query and the response are under the max
    length threshold
    """
    return [pair for pair in pairs if isPairLengthValid(pair, max_length)]


# --- VOCABULARY / PAIRS CONSTRUCTION METHODS

def readIntoVoc(datafile_path: str, corpus_name: str) -> Tuple[Voc, List]:
    """
    Read query response pairs from file and return a Voc object
    """
    # Read the file and split into lines
    lines = open(datafile_path, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    return Voc(corpus_name), pairs


def trimRareWords(voc: Voc, pairs: List, min_count: int):
    """
    Remove rare words from the vocabulary and pairs
    """
    def should_keep(sentence: str) -> bool:
        """
        True if all the words in the sentence are in the vocabulary.
        """
        for word in sentence.split(' '):
            if word not in voc.word2index:
                return False
        return True

    voc.trim(min_count)
    # Filter out pairs containing the trimmed words
    keep_pairs = list()
    for pair in pairs:
        input_sentence, output_sentence = pair
        # Keep pairs where trimmed words are not used in both input and output.
        if should_keep(input_sentence) and should_keep(output_sentence):
            keep_pairs.append(pair)
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(
        len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def loadPrepareDataset(
        corpus_name: str, datafile_path: str,
        dull_responses: List[str], min_count: int = 3, max_length: int = 15):
    """
    Load the dataset and add the pairs as well as the dull responses.
    """
    print("Start preparing training data ...")
    voc, pairs = readIntoVoc(datafile_path, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    for dull_response in dull_responses:
        voc.addSentence(dull_response)
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    # Remove very rare words from vocabulary and pairs.
    pairs = trimRareWords(voc, pairs, min_count)
    return voc, pairs


# ---- LOSS METHODS

def maskNLLLoss(device, inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = - \
        torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# ---- MODEL CONSTRUCTION METHODS


def inputVar(l: List, voc: Voc) -> Tuple:
    """
    Given a list of sentences and vocabulary, constructs the input variable.
    Returns the padded variables and the lengths of the vairables.
    """
    indices_batch = [voc.getIndicesFromSentence(sentence) for sentence in l]
    lengths = torch.tensor([len(indices) for indices in indices_batch])
    padList = zeroPadding(indices_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVar(l: List[str], voc: Voc) -> Tuple:
    """
    Given a list of sentences and the vocabulary, returns the output variable
    and the lengths of the variables.
    """
    indices_batch = [voc.getIndicesFromSentence(sentence) for sentence in l]
    max_target_len = max([len(indices) for indices in indices_batch])
    padList = zeroPadding(indices_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batchToTrainData(voc: Voc, pair_batch: List):
    """
    Given a vocabulary and batch of pairs, obtain the input variables,
    lengths, output variables, mask, and max target length that are
    needed for the encoder-decoder.
    """
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = list(), list()
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


def loadModel(voc: Voc, saved_model_path: str, embedding: nn.Embedding,
              encoder: nn.Module, decoder: nn.Module, encoder_optimizer,
              decoder_optimizer) -> Dict:
    """
    Method for loading the encoder / decoder model
    """
    checkpoint = {'iteration': 0}
    # Load model from previous checkpoint
    if os.path.isfile(saved_model_path):
        checkpoint = torch.load(
            saved_model_path, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
        embedding.load_state_dict(embedding_sd)
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)
    return checkpoint


def generateSavePath(save_dir, model_name, corpus_name, encoder_n_layers,
                     decoder_n_layers, hidden_size):
    return os.path.join(
        save_dir, model_name, corpus_name,
        "{}-{}_{}".format(
            encoder_n_layers, decoder_n_layers, hidden_size))


def saveModel(save_dir, model_name, corpus_name, encoder_n_layers,
              decoder_n_layers, hidden_size, iteration, encoder, decoder,
              encoder_optimizer, decoder_optimizer, loss, voc, embedding):
    directory = generateSavePath(
        save_dir, model_name, corpus_name, encoder_n_layers,
        decoder_n_layers, hidden_size)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({
        'iteration': iteration,
        'en': encoder.state_dict(),
        'de': decoder.state_dict(),
        'en_opt': encoder_optimizer.state_dict(),
        'de_opt': decoder_optimizer.state_dict(),
        'loss': loss,
        'voc_dict': voc.__dict__,
        'embedding': embedding.state_dict()
    }, os.path.join(directory, '{}_{}.tar'.format(
        iteration, 'checkpoint')))


# ---- RL Methods


def set_optimizer_learning_rate(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def transformTensorToSameShapeAs(tensor, shape):
    """
    Obtain a new tensor of the the given shape and the associated mask.
    """
    size1, size2 = shape
    npNewT = np.zeros((size1, size2), dtype=np.int64)
    npNewMask = np.zeros((size1, size2), dtype=np.bool_)
    tensorSize1, tensorSize2 = tensor.size()
    for i in range(tensorSize1):
        for j in range(tensorSize2):
            npNewT[i][j] = tensor[i][j]
            npNewMask[i][j] = True
    return torch.from_numpy(npNewT), torch.from_numpy(npNewMask)


def convertResponse(response, batch_size):
    # NOTE: Batch_size was not being passed in here.
    size1 = len(response)
    size2 = batch_size
    npRes = np.zeros((size1, size2), dtype=np.int64)
    npLengths = np.zeros(size2, dtype=np.int64)
    for i in range(size1):
        prov = response[i].cpu().numpy()
        for j in range(prov.size):
            npLengths[j] = npLengths[j] + 1
            if prov.size > 1:
                npRes[i][j] = prov[j]
            else:
                npRes[i][j] = prov
    res = torch.from_numpy(npRes)
    lengths = torch.from_numpy(npLengths)
    return res, lengths


def convertTarget(target, batch_size):
    # NOTE: Batch_size was not being passed in here.
    size1 = len(target)
    size2 = batch_size
    npRes = np.zeros((size1, size2), dtype=np.int64)
    mask = np.zeros((size1, size2), dtype=np.bool_)
    npLengths = np.zeros(size2, dtype=np.int64)
    for i in range(size1):
        prov = target[i].cpu().numpy()
        for j in range(prov.size):
            npLengths[j] = npLengths[j] + 1
            if prov.size > 1:
                npRes[i][j] = prov[j]
            else:
                npRes[i][j] = prov

            if npRes[i][j] > 0:
                mask[i][j] = True
            else:
                mask[i][j] = False

    res = torch.from_numpy(npRes)
    lengths = torch.from_numpy(npLengths)
    mask = torch.from_numpy(mask)
    max_target_len = torch.max(lengths)  # .detach().numpy()
    return res, mask, max_target_len


# ------ EVALUATION METHODS


def evaluate(searcher, voc, sentence, max_length, device):
    indices_batch = [voc.getIndicesFromSentence(sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indices) for indices in indices_batch])
    lengths = lengths.to('cpu')
    # Transpose dimensions of batch to match model's expectations
    input_batch = torch.LongTensor(indices_batch).transpose(0, 1)
    # Use the appropriate device
    input_batch = input_batch.to(device)
    # lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, _ = searcher(input_batch, lengths, max_length)
    # Indices to words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(searcher, voc, max_length, device):
    while True:
        try:
            # Get input sentence
            input_sentence = input('> ')
            if input_sentence.lower() == 'q' or input_sentence.lower() == 'quit':
                break
            # Normalize the strings
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(
                searcher, voc, input_sentence, max_length, device)
            # Format and print response
            output_words[:] = [
                x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Agent: {}'.format(' '.join(output_words)))
        except Exception as e:
            print(e)

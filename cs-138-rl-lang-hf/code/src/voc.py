from typing import List, Dict
# from collections import defaultdict
from copy import deepcopy
from .vars import *


class Voc:
    """
    Stores the vocabulary that is being used.
    """

    def __init__(self, vocabulary_name: str):

        self.name = vocabulary_name
        self.trimmed = False
        self.word2index = dict()
        self.word2count = dict()
        self.index2word = {
            PAD_TOKEN: 'PAD',
            SOS_TOKEN: 'SOS',
            EOS_TOKEN: 'EOS'}
        self.num_words = 3  # Number of the tokens
        self.info = {
            "words_added": 0,
            "sentences_added": 0,
            "trimmed_words": 0,
            "kept_words": 0
        }

    def addSentence(self, sentence: str) -> None:
        """
        Add each word from a sentence to the vocabulary.
        """
        for word in sentence.split(' '):
            self.addWord(word)
        # Update count
        self.info["sentences_added"] += 1

    def addWord(self, word: str) -> None:
        """
        Add a word to the vocabulary.
        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
            self.info["words_added"] = self.num_words
        else:
            self.word2count[word] += 1

    def trim(self, min_count: int) -> None:
        """
        Remove with a count below the given threshold.
        Returns the percentage of words kept.
        """
        if self.trimmed:
            return
        keep_words = [k for k, v in self.word2count.items() if v >= min_count]
        # Update the info dict.
        self.info["trimmed_words"] = len(self.word2index) - len(keep_words)
        self.info["kept_words"] = len(keep_words)

        # Reset dictionaries and add words again
        self.word2index = dict()
        self.word2count = dict()
        self.index2word = {
            PAD_TOKEN: 'PAD',
            SOS_TOKEN: 'SOS',
            EOS_TOKEN: 'EOS'}
        self.num_words = 3  # Number of the tokens
        for word in keep_words:
            self.addWord(word)
        self.trimmed = True
        # return len(keep_words) / len(self.word2index)

    def getIndicesFromSentence(self, sentence: str) -> List:
        """
        Given a sentence, obtain a list of indices for the corresponding to the
        indices of the words in the vocabulary.
        """
        return [self.word2index[word] for word in sentence.split(' ')] \
            + [EOS_TOKEN]

    def get_info(self) -> Dict:
        """
        Obtain information about the dictionary.
        """
        return deepcopy(self.info)

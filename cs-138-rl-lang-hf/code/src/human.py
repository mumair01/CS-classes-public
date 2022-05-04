# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2021-11-30 09:19:04
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2021-11-30 10:08:04
# @desc Contains the human evaluation methods

# Standard imports
from typing import List
# Local imports
from .voc import Voc
from .vars import *
import numpy as np

class HumanTrainer:

    def __init__(self, human_metrics: List[str], query_every: int, voc: Voc):
        self.human_metrics = human_metrics
        self.query_every = query_every
        self.voc = voc
        self.curr_iteration = 0
        self.min_rating = 0
        self.max_rating = 10
        self.separator = "=" * 20
        self.maxBaselineRating = 50
        self.num_responses_shown = 10
        self.curr_reward = 0

    def step(self, forward_encoder_optimizer, forward_decoder_optimizer):
        """
        Updates the internal count of the trainer such that it can query
        every N iterations.
        """
        self.curr_iteration += 1

        if (self.curr_iteration % self.query_every) == 0:
            for g in forward_encoder_optimizer.param_groups:
                g['lr'] = 0.01   
            for g in forward_decoder_optimizer.param_groups:
                g['lr'] = 0.01  
            return True
        
        else:
            for g in forward_encoder_optimizer.param_groups:
                g['lr'] = 0.0001    
            for g in forward_decoder_optimizer.param_groups:
                g['lr'] = 0.0001  
            return False
        

    def query_trainer(self, statements, responses):
        """
        Query the trainer given the statements and their responses vectors.
        """
        print("Querying human trainer...")
        
        # Print a sample of human responses in a viewable format
        formated_responses = []
        arr = [i.tolist() for i in responses]
        arr = np.asarray(arr).squeeze().transpose()
        (sentence_num, sentence_len) = arr.shape

        # Add to formated responses
        for sentence_index in range(sentence_num):
            sentence = ""
            for word_index in range(sentence_len):
                word = self.voc.index2word[arr[sentence_index][word_index]]
                if word != 'EOS':
                    sentence += word + ' '
            formated_responses.append(sentence)  
        formated_responses = formated_responses[:self.num_responses_shown]
    
        # Show and query human feedback on categories
        print(self.separator)
        for res in formated_responses:
            print(res)
        print(self.separator)
        print("\nPlease answer the following on a scale of [" + str(self.min_rating) + ", " + str(self.max_rating) + "]")
        
        # Update the reward
        self.curr_reward = 0
        for metric in self.human_metrics:
            # self.curr_reward += ((float(input(metric))) - self.num_responses_shown) * (-1 * self.human_reward_scaling_factor)
            self.curr_reward += (float(input(metric)))
        self.curr_reward = (self.curr_reward / len(self.human_metrics) / self.max_rating) * self.maxBaselineRating
                       
    def update(self):
        """
        Update the learning rates of the given encoder, decoder, and output
        new reward using internal state.
        """
        print("Updating the reward...")
        return self.curr_reward
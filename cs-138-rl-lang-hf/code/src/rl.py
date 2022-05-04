# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2021-11-29 18:23:29
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2021-11-30 13:28:39

# Third party imports
import random
import os
import torch
import numpy as np
import src.utils as utils
from .voc import Voc
from .vars import *
from .train import step
from .human import HumanTrainer
from scipy.spatial import distance


# -------- RL Reward methods

# TODO: This should operate at the word level somehow.
def easeOfAnswering(
        device, voc, input_variable, lengths, dull_responses, max_target_len,
        encoder, decoder, batch_size, teacher_forcing_ratio):
    r1 = 0
    return r1
    for d in dull_responses:
        d, _, max_target_len = utils.outputVar(d, voc)  # TODO: Error here.
        newD, newMask = utils.transformTensorToSameShapeAs(
            d, input_variable.size())
        # Get the forward loss
        _, forward_loss, forward_len, _ = step(
            device, encoder, decoder, input_variable, lengths, newD,
            newMask, max_target_len, batch_size, teacher_forcing_ratio)
        if forward_loss > 0:
            r1 -= forward_loss / forward_len
    return r1 / len(dull_responses) if len(dull_responses) > 0 else r1

# NOTE: What are responses?


def informationFlow(responses):
    if len(responses) <= 2:
        return 0
    h_pi = responses[-3]
    h_pi1 = responses[-1]
    min_length = min(len(h_pi), len(h_pi+1))
    h_pi = h_pi[:min_length]
    h_pi1 = h_pi1[:min_length]
    cos_sim = 1 - \
        distance.cdist(h_pi.cpu().numpy(), h_pi1.cpu().numpy(), 'cosine')
    return np.mean(- cos_sim if np.any(cos_sim <= 0) else - np.log(cos_sim))

# TODO: What are responses?


def informationFlow(responses):
    r2 = 0
    if len(responses) > 2:  # NOTE: Why is 2 arbitrarily being used here?
        h_pi = responses[-3]
        h_pi1 = responses[-1]
        min_length = min(len(h_pi), len(h_pi+1))
        h_pi = h_pi[:min_length]
        h_pi1 = h_pi1[:min_length]
        cos_sim = 1 - \
            distance.cdist(h_pi.cpu().numpy(), h_pi1.cpu().numpy(), 'cosine')
        if np.any(cos_sim <= 0):
            r2 = - cos_sim
        else:
            r2 = - np.log(cos_sim)
        r2 = np.mean(r2)
    return r2


def semanticCoherence(
        device, input_variable, lengths, target_variable, mask,
        max_target_len, forward_encoder, forward_decoder,
        backward_encoder, backward_decoder, batch_size,
        teacher_forcing_ratio):
    r3 = 0
    # NOTE: Again, why are we using the whole RL function here???
    _, forward_loss, forward_len, _ = step(
        device, forward_encoder, forward_decoder, input_variable, lengths,
        target_variable,  mask, max_target_len, batch_size, teacher_forcing_ratio)
    # TODO: This seems switched. Shouldn't response go to ConvertResponse and
    # vice versa?
    ep_input, lengths_trans = utils.convertResponse(
        target_variable, batch_size)
    ep_target, mask_trans, max_target_len_trans = utils.convertTarget(
        input_variable, batch_size)
    _, backward_loss, backward_len, _ = step(
        device, backward_encoder, backward_decoder, ep_input, lengths_trans,
        ep_target,  mask_trans, max_target_len_trans, batch_size,
        teacher_forcing_ratio)
    if forward_len > 0:
        r3 += forward_loss / forward_len
    if backward_len > 0:
        r3 += backward_loss / backward_len
    return r3


def calculateRewards(
        device, voc, input_var, lengths, target_var, mask, max_target_len,
        forward_encoder, forward_decoder, backward_encoder,
        backward_decoder, batch_size, teacher_forcing_ratio,
        min_count, dull_responses, max_length, num_episodes=10):
    # Initializations
    ep_rewards = list()  # Rewards per episode
    responses = list()  # List of responses
    ep_input = input_var  # Input of the current episode
    ep_target = target_var  # Target of the current episode
    for _ in range(num_episodes):
        _, _, _, curr_response = step(
            device, forward_encoder, forward_decoder, ep_input, lengths,
            ep_target, mask, max_target_len, batch_size, teacher_forcing_ratio)
        if len(curr_response) < min_count:
            break
        # Calculate the different reward metrics
        r1 = easeOfAnswering(
            device, voc, ep_input, lengths, dull_responses, max_target_len,
            forward_encoder, forward_decoder, batch_size, teacher_forcing_ratio)
        r2 = informationFlow(responses)
        r3 = semanticCoherence(
            device, ep_input, lengths, target_var, mask, max_target_len,
            forward_encoder, forward_decoder, backward_encoder, backward_decoder,
            batch_size,  teacher_forcing_ratio)
        # Combine the rewards using weights to create final reward.
        r = 0.25 * r1 + 0.25 * r2 + 0.5 * r3
        # Add the current rewards to the rewards list
        ep_rewards.append(r.detach().cpu().numpy())
        # Add the responses to the response list.
        curr_response, lengths = utils.convertResponse(
            curr_response, batch_size)
        curr_response = curr_response.to(device)
        responses.append(curr_response)
        # Use the next input as the current response
        ep_input = curr_response
        ep_target = torch.zeros(max_length, batch_size, dtype=torch.int64)
        ep_target = ep_target.to(device)
        # Turn off teacher forcing after first interaction
        teacher_forcing_ratio = 0
    # Take the mean of the episodic rewards
    return np.mean(ep_rewards) if len(ep_rewards) > 0 else 0


# -------- RL Training Methods


def rl_train(device, forward_encoder, forward_encoder_optimizer, forward_decoder,
             forward_decoder_optimizer, backward_encoder, backward_encoder_optimizer,
             backward_decoder, backward_decoder_optimizer, voc, input_variable,
             lengths, target_variable, mask, max_target_len, batch_size,
             teacher_forcing_ratio, dull_responses, min_count, max_length,
             human_trainer: HumanTrainer):
    # The optimizers are cumulative and have to be zeroed.
    forward_encoder_optimizer.zero_grad()
    forward_decoder_optimizer.zero_grad()
    backward_encoder_optimizer.zero_grad()
    backward_decoder_optimizer.zero_grad()
    # Step to generate loss and length
    _, forward_loss, forward_len, responses = step(
        device, forward_encoder, forward_decoder, input_variable, lengths,
        target_variable,
        mask, max_target_len, batch_size, teacher_forcing_ratio)
    # Calculate the rewards
    reward = calculateRewards(
        device, voc, input_variable, lengths, target_variable, mask,
        max_target_len, forward_encoder, forward_decoder,
        backward_encoder, backward_decoder, batch_size,
        teacher_forcing_ratio, min_count, dull_responses, max_length)
    # If time to query human
    if human_trainer.step(forward_encoder_optimizer, forward_decoder_optimizer):
        # Replace the reward with the corresponding human feedback
        human_trainer.query_trainer(input_variable, responses)
        reward = human_trainer.update()
    # Calculate the loss scaled by the reward and backpropagate
    loss = forward_loss * reward
    loss.backward()
    forward_encoder_optimizer.step()
    forward_decoder_optimizer.step()
    return loss, forward_len, reward

def rl_iters(
        device, forward_encoder, forward_encoder_optimizer, forward_decoder,
        forward_decoder_optimizer, backward_encoder, backward_encoder_optimizer,
        backward_decoder, backward_decoder_optimizer, voc, pairs, batch_size,
        teacher_forcing_ratio, dull_responses, n_iterations, print_every,
        save_every, query_every, save_dir, checkpoint, loadFilename,
        min_count, max_length, model_name, corpus_name, encoder_n_layers,
        hidden_size, decoder_n_layers, embedding, human_trainer: HumanTrainer):

    # Load batches for training iterations
    training_batches = [utils.batchToTrainData(
        voc, [random.choice(pairs) for _ in range(batch_size)])
        for _ in range(n_iterations)]
    # Initializing
    print("Initializing RL training...")
    print_loss = 0
    start_iteration = checkpoint['iteration'] + 1 \
        if os.path.isfile(loadFilename) else 1
    # Training loop
    print("Starting RL training loop...")
    for iteration in range(start_iteration, n_iterations + 1):
        training_batch = training_batches[iteration-1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = \
            training_batch
        # Train using rl mechanism
        loss, forward_len, reward = rl_train(
            device, forward_encoder, forward_encoder_optimizer, forward_decoder,
            forward_decoder_optimizer, backward_encoder, backward_encoder_optimizer,
            backward_decoder, backward_decoder_optimizer, voc,
            input_variable, lengths, target_variable, mask, max_target_len,
            batch_size, teacher_forcing_ratio, dull_responses, min_count,
            max_length, human_trainer)

        # Print the loss if needed
        print_loss += loss / forward_len
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete:"
                  " {:.2f}%; Average loss: {:.4f}; Reward: {:.2f}".format(
                      iteration, iteration / n_iterations * 100,
                      print_loss_avg, reward))
            print_loss = 0
        # Save the model
        if iteration % save_every == 0:
            utils.saveModel(
                save_dir, model_name, corpus_name, encoder_n_layers,
                decoder_n_layers, hidden_size, iteration, forward_encoder,
                forward_decoder, forward_encoder_optimizer,
                forward_decoder_optimizer, loss, voc, embedding)

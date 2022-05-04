'''
This script contains training methods for training the Seq2Seq model.
'''
# Local imports
import random
import torch
import torch.nn as nn
from .vars import *
import src.utils as utils
import os


def step(
        device, encoder, decoder, input_variable, lengths, target_variable,
        mask, max_target_len, batch_size, teacher_forcing_ratio=0.5):
    """
    Generate a series of responses from the given inputs and return some metrics.
    """
    # Set the device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    lengths = lengths.to('cpu')
    # Initialize variables
    loss = n_totals = 0
    print_losses = list()
    responses = list()
    # Forward through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    # Create initial decoder input
    decoder_input = torch.LongTensor(
        [[utils.SOS_TOKEN for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    # Set initial decoder hidden state to encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs)

        if random.random() < teacher_forcing_ratio:
            # Case 1: Use teacher forcing.
            decoder_input = target_variable[t].view(1, -1)
        else:
            # Case 2: Do not use teacher forcing
            # Return the k largest elements of tensor on given dimension
            # NOTE: We are getting the top indices here.
            _, topi = decoder_output.topk(1)

            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            responses.append(topi)
        # Calculate and accumulate the loss.
        mask_loss, nTotal = utils.maskNLLLoss(
            device, decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal
    # NOTE: What is this print_loss???
    print_loss = sum(print_losses) / n_totals
    return print_loss, loss, max_target_len, responses


def train(device, encoder, decoder, encoder_optimizer, decoder_optimizer,
          input_variable, lengths, target_variable, mask, max_target_len,
          batch_size, clip,  teacher_forcing_ratio=0.5):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    print_loss, loss, _, _ = step(
        device, encoder, decoder, input_variable, lengths, target_variable,
        mask, max_target_len, batch_size, teacher_forcing_ratio)
    # Perform backpropagation
    loss.backward()
    # Clip gradients in-place to solve the exploding gradient problem
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    # Adjust the model weights
    encoder_optimizer.step()
    decoder_optimizer.step()
    return print_loss


def train_iters(device, encoder, decoder, encoder_optimizer, decoder_optimizer,
                voc, pairs, batch_size, n_iterations, loadFilename,
                checkpoint, clip, print_every, save_every, save_dir, model_name,
                corpus_name, encoder_n_layers, decoder_n_layers, hidden_size,
                embedding):
    # Load batches for training iterations
    training_batches = [utils.batchToTrainData(
        voc, [random.choice(pairs) for _ in range(batch_size)])
        for _ in range(n_iterations)]
    # Initializing
    print("Initializing training...")
    print_loss = 0
    start_iteration = checkpoint['iteration'] + 1 \
        if os.path.isfile(loadFilename) else 1
    # Training loop
    print("Starting training loop...")
    for iteration in range(start_iteration, n_iterations + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from the batch
        input_variable, lengths, target_variable, mask, max_target_len = \
            training_batch
        # Run a training iteration with the given batch
        loss = train(
            device, encoder, decoder, encoder_optimizer, decoder_optimizer,
            input_variable, lengths, target_variable, mask, max_target_len,
            batch_size, clip)
        print_loss += loss
        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print('Iteration {}; Percentage complete {:.1f}%; Average loss: {:.4f}'.format(
                iteration, iteration / n_iterations * 100, print_loss_avg))
            print_loss = 0
        # Save checkpoint
        if iteration % save_every == 0:
            utils.saveModel(
                save_dir, model_name, corpus_name, encoder_n_layers,
                decoder_n_layers, hidden_size, iteration, encoder, decoder,
                encoder_optimizer, decoder_optimizer, loss, voc, embedding)
    print("Completed training!")

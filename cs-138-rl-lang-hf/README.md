# Natural Language Generation using Reinforcement Learning with Human Feedback

## About

This repository contains the proof of concept implementation of using reinforcement learning with human feecback to shape language models.

**NOTE**: This was presented as the final project for CS-138: Reinforcement Learning, Fall 2021, at Tufts University. Partners for this project were Miles Izydorczak, Kevin Gao.

## Usage

Code Structure:

- chatbot folder - driver function for the chatbot.
- src folder - implementation of RL policies, neural models, reward metrics, human querying, etc.
- data - not included b/c file sizes, but can be downloaded online @ https://opus.nlpl.eu/OpenSubtitles-v2018.php

Install dependencies:

`pip install -r ./code/requirements.txt`

Running Code:

- This code requires jupyter notebook and the RL and neural models are implementing in PyTorch.
- To run, navigate to Chatbot.ipynb in ./chatbot and run the notebook.

## Overview

Conversational agents are a widespread means of providing customer
service in commercial domains, yet studies have shown that consumers of-
ten struggle to benefit from these interactions, and similar difficulties
arise for open-domain chatbots. In this paper, we build upon previous work on Maximum Likelihood Estimation-based natural language
models by incorporating customizable forward-looking reward functions
through reinforcement learning and human shaping to produce conversational agents that are capable of optimizing for future rewards, and we
provide our framework in a generalizable structure that is applicable to
both open-domain and closed-domain chatbots. Our results show that
language models trained with reinforcement learning and human feedback are capable of producing more coherent, diverse, and progressive
conversations.

## Architecture Overview

<p align="center">
<image src="resources/framework.png" width="80%" height="80%">
</p>

Figure 1 shows the architecture of our generalizable framework for incorporating human feedback and customizable reward metrics into the learning process. There are four main components: the language model, the RL training algorithm, the baseline reward metrics, and the human rewards. The language model is an encoder-decoder RNN, which is pre-trained on dialogue samples from the open-subtitles dataset. Once the language model is pre-trained, we use two copies of the language model to create two agents for that take turns in the RL training process producing utterances for which a reward can be calculated using the baseline metrics and human feedback.

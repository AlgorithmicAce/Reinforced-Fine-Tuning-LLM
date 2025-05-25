# Reinforced-Fine-Tuning-LLM using GRPO
This repo contains a Python script used to fine-tune open-source LLMs to carry out reasoning tasks like playing Wordle

Reasoning LLMs such as DeepSeek R1, ChatGPT o1, Gemini 2.5, Claude 3.7 Sonnet Thinking performs better than general LLMs in carrying out tasks like Wordle.
> Wordle is a game where the user has to guess the 5-letter word under 6 tries. For each trial, the guess gets feedback. If a letter is correct and in correct position, it gets a check (✔). If a letter is correct but in the wrong position, then it gets a dash (-). If a letter is wrong, then it gets a cross (❌).

How is **Group Relative Policy Optimization (GRPO)** differe from other policy **Reinforcement Learning with Human Feedback (RLHF)** and **Direct Policy Optimization (DPO)**
> RLHF & DPO need a huge dataset of human preferences (Which response does the user like more) --> Not efficient
> GRPO uses reward-based learning (Reward from pre-defined functions or LLM as a judge) and uses the reward  to update the weights to maximise reward

I'm fine-tuning **Gemma 3 1B** using [Unsloth AI](https://docs.unsloth.ai/) since it allows quantised fine-tuning. So rather than using industrial grade GPU with higher VRAM, I can fine-tune on Google Colab with just 15GB VRAM.

 **Progress:**
 

 - [x] Fine-tuned using OpenAI's famous **gsm8k** dataset
 - [x] Created reward functions for **Wordle GRPO** dataset
 - [ ] Fine-tuned **Gemma 3 1B** (Training takes hours and Colab only offers 3-4 hours of Tesla T4 GPU) --> Can try using Kaggle
 - [ ] Uploaded model to HuggingFace

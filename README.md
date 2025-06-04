# Reinforced-Fine-Tuning-LLM using GRPO
This repo contains a Python script used to fine-tune open-source LLMs to carry out reasoning tasks like playing Wordle

Reasoning LLMs such as DeepSeek R1, ChatGPT o1, Gemini 2.5, Claude 3.7 Sonnet Thinking perform better than general LLMs in carrying out tasks like Wordle.
> Wordle is a game where the user has to guess the 5-letter word under 6 tries. For each trial, the guess gets feedback. If a letter is correct and in the correct position, it gets a check (✔). If a letter is correct but in the wrong position, then it gets a dash (-). If a letter is wrong, then it gets a cross (❌).

## **GRPO Algorithm**
At first glance, the GRPO Algorithm looks complicated. But we can break down the formula into 4 parts
Complete algorithm:
![alt text][complete algorithm]

1. Policy Loss
   ![alt text][policy loss]
2. Advantage
   ![alt text][advantage]
3. Clipping Objective
   ![alt text][clipping objective]
4. KL Divergence
   ![alt text][kl divergence]

> So, what this formula does is it takes the ratio of token probability distributions in your model with and without the adapter and multiplies it by the advantage. Clipping is used ot make sure we don't have a large loss value for any individual step and then multiply by that advantage. Min function is used to choose the minimum value, either the unclipped ratio or the clipped ratio. Then, finally, KL divergence is subtracted from the minimum ratio. This is to make sure the model we're training doesn't deviate too much from the baseline knowledge that it already knows

How is **Group Relative Policy Optimization (GRPO)** differe from other policy **Reinforcement Learning with Human Feedback (RLHF)** and **Direct Policy Optimization (DPO)**
> RLHF & DPO need a huge dataset of human preferences (Which response does the user like more) --> Not efficient
> GRPO uses reward-based learning (Reward from pre-defined functions or LLM as a judge) and uses the reward  to update the weights to maximise reward

I'm fine-tuning **Qwen 2.5 3B** using [Unsloth AI](https://docs.unsloth.ai/) since it allows quantised fine-tuning. So, rather than using an industrial-grade GPU with higher VRAM, I can fine-tune on Google Colab with just 16GB VRAM.
I was using **Gemma 3 1B** model previously, but even with different prompts, I couldn't let the model "understand" the task! It keeps hallucinating, and I felt like **Qwen 2.5 3B** is better so I chose to proceed with that model

**Reward Functions For Wordle:**
1. Output Format Check: Checks if the output matches the exact format
2. Uses Previous Feedback: Checks if the output is using previous feedback
3. Guess Value: Rewards the word that it guessed

 **Progress:**
 - [x] Fine-tuned using OpenAI's famous **gsm8k** dataset
 - [x] Created reward functions for **Wordle GRPO** dataset
 - [ ] Fine-tuned **Qwen 2.5 3B** (Training takes hours and Colab only offers 3-4 hours of Tesla T4 GPU) --> Can try using Kaggle --> I'm using university PC with RDP
 - [ ] Uploaded model to HuggingFace

[complete algorithm]: https://github.com/AlgorithmicAce/Reinforced-Fine-Tuning-LLM/blob/main/images/GRPO.jpeg "Group Relative Policy Optimisation"
[policy loss]: https://github.com/AlgorithmicAce/Reinforced-Fine-Tuning-LLM/blob/main/images/Policy%20Loss.jpeg "Represents the ratio of token probability distributions in your model with and without adapter"
[advantage]: https://github.com/AlgorithmicAce/Reinforced-Fine-Tuning-LLM/blob/main/images/Advantage.jpeg "Advantages are rewards that are normalised to be centred around 0"
[clipping objective]: https://github.com/AlgorithmicAce/Reinforced-Fine-Tuning-LLM/blob/main/images/Clip.jpeg "Used to make sure that we don't have a large loss value for any individual step"
[kl divergence]: https://github.com/AlgorithmicAce/Reinforced-Fine-Tuning-LLM/blob/main/images/KL%20Divergence.jpeg "Used to make sure the model we're training doesn't deviate too much from the baseline knowledge that it already knows"

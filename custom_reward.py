import numpy as np
import torch
from utils import *
import warnings
warnings.filterwarnings("ignore")

model_id = "google/gemma-3-4b-it"

# Binary reward function (If the guessed_word == secret word, then 1, else 0)
def wordle_binary_reward(guess: str, secret_word: str):
    if guess.upper() == secret_word.upper(): # In case there's some capitalization issues
        return 1
    else:
        return 0
    
secret_word = "POUND"

past_guesses = [
    GuessWithFeedback.from_secret(guess="CRANE", secret = secret_word),
    GuessWithFeedback.from_secret(guess="BLOND", secret = secret_word),
    GuessWithFeedback.from_secret(guess="FOUND", secret = secret_word)
]

#print(past_guesses)

response = generate(get_messages(past_guesses))[0] # Model response
guess = extract_guess(response) # Extract the guessed word using regex
reward = wordle_binary_reward(guess, secret_word) # Calculates the binary reward value

# print(f"Guessed Word: {guess} -> Reward: {reward}")

def compute_advantages(rewards):
    rewards = np.array(rewards)
    # Advantage is a normalized reward which is centered around 0

    # Compute mean and std dev
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # If all values are the same, std dev (sigma) will be 0, causing Division By Zero Error
    if std_reward == 0:
        return [0] * len(rewards)
    
    # Normalized the rewards
    advantages = (rewards - mean_reward) / std_reward
    return advantages.tolist()

def render_guess_table(response, reward_fn):
    guesses = [extract_guess(guess) for guess in response]
    rewards = [reward_fn(guess, secret_word) for guess in guesses]
    print_guesses_table(guesses, rewards)

print(f"Secret: {secret_word}")
response = generate(get_messages(past_guesses), num_guesses=8)
#print(response)
render_guess_table(response, wordle_binary_reward)

# String "BOUND" has reward value of 0, while being so close to the secret word ("POUND")
# String "CRANE" has reward value of 0 as well, so the model doesn't know the diffference between BOUND & CRANE compared to POUND
# So we are going to create a function that gives partial rewards

def wordle_reward_partial_credit(guess: str, secret_word: str):
    if len(guess) != len(secret_word):
        return 0.0
    valid_letters = set(secret_word) #removes double letters
    reward = 0.0
    for letter, secret_letter in zip(guess, secret_word):
        if letter == secret_letter:
            reward += 0.2
        elif letter in valid_letters:
            reward += 0.1
        else:
            pass
    return reward

print(f"Secret: {secret_word}")
response = generate(get_messages(past_guesses), num_guesses=8, temperature=0)
render_guess_table(response, wordle_reward_partial_credit)

print(f"Secret: {secret_word}")
response = generate(get_messages(past_guesses), num_guesses=8, temperature=1.3)
render_guess_table(response, wordle_reward_partial_credit)

print(f"Secret: {secret_word}")
response = generate(get_messages(past_guesses), num_guesses=8, temperature=1.0)
render_guess_table(response, wordle_reward_partial_credit)
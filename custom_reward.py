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

print(f"Guessed Word: {guess} -> Reward: {reward}")
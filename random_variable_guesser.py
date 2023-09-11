import math
from dataclasses import dataclass
from functools import partial, reduce
from typing import Sequence

# wrap RandomVariable so log probabilities and type specs are more explicit

@dataclass
class RandomVariable:
    log_probabilities: dict

def softmax_combine(log_x, log_y):
    return log_x + math.log1p(math.exp(log_y - log_x))

def log_expectation(key_to_log_val_func, random_var: RandomVariable): # inject function, this is like log(E[f(X)])
    if not random_var.log_probabilities:
        return -math.inf
    return reduce(softmax_combine,
                    (log2_prob + key_to_log_val_func(key) for key, log2_prob in random_var.log_probabilities.items())
                    )

# wrap return for more readable and cohesive output
 
@dataclass(frozen=True)
class Guess:
    log_expected_probability: float = 0.0
    code: tuple[int] = tuple()
    

def max_log_expected_probability_guess(clue_and_keyword_to_log_prob_func, random_vars: Sequence[RandomVariable], clues: tuple[str]) -> Guess:
    # best_guess[mask][i] = max log expectation guess for random_vars specified by mask and clues[:i]
    best_guess = [[Guess()] * (len(clues) + 1) for _ in range(1 << len(random_vars))]
    
    for clue_index, clue in enumerate(clues):
        # curry the clue to make a first argument for log_expectation
        keyword_to_log_prob_given_clue_func = partial(clue_and_keyword_to_log_prob_func, clue)

        # enumerate each nonempty subset of random variables
        for var_indices in range(1, 1 << len(random_vars)):
            
            # compute the clue heuristic for each available random variable
            guesses = []
            for var_index, random_var in enumerate(random_vars):
                if var_indices & (1 << var_index) == 0:
                    continue
                remaining_var_indices = var_indices - (1 << var_index)
                subproblem_best_guess = best_guess[remaining_var_indices][clue_index]
                log_expected_probability = subproblem_best_guess.log_expected_probability + log_expectation(keyword_to_log_prob_given_clue_func, random_var)
                guess = Guess(log_expected_probability, subproblem_best_guess.code + (var_index,))
                guesses.append(guess)
            
            # choose the random variable with the best clue heuristic
            best_guess[var_indices][clue_index + 1] = max(guesses, key = lambda guess: guess.log_expected_probability)

    all_random_vars = (1 << len(random_vars)) - 1
    return best_guess[all_random_vars][len(clues)]
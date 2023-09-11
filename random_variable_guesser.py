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
    
    num_random_var_subsets = 1 << len(random_vars)
    # best_guess[bitmask] = max log expectation guess for random_vars specified by bitmask 
    best_guess = [Guess()] * num_random_var_subsets # when no clues have been considered, the best guess is an empty code with probability 1
    
    # update best_guess with each clue
    for clue in clues:
        # curry the clue to make a first argument for log_expectation
        keyword_to_log_prob_given_clue_func = partial(clue_and_keyword_to_log_prob_func, clue)

        best_guess_with_clue = [Guess(-math.inf)] * num_random_var_subsets
        
        # enumerate each nonempty subset of random variables
        for vars_bitmask in range(1, num_random_var_subsets):
            
            # compute the clue heuristic for each available random variable
            guesses = []
            for var_index, random_var in enumerate(random_vars):
                if vars_bitmask & (1 << var_index) == 0:
                    continue

                remaining_var_indices = vars_bitmask - (1 << var_index)
                subproblem_best_guess = best_guess[remaining_var_indices]
                log_expected_probability = subproblem_best_guess.log_expected_probability + log_expectation(keyword_to_log_prob_given_clue_func, random_var)
                guess = Guess(log_expected_probability, subproblem_best_guess.code + (var_index,))
                guesses.append(guess)
            
            # choose the guess with the best clue heuristic
            best_guess_with_clue[vars_bitmask] = max(guesses, key = lambda guess: guess.log_expected_probability)
        
        best_guess = best_guess_with_clue

    all_random_vars = num_random_var_subsets - 1
    return best_guess[all_random_vars]
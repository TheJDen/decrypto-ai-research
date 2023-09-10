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
        

def max_expected_log_probability_guess(clue_and_keyword_to_log_prob_func, random_vars: Sequence[RandomVariable], clues: tuple[str]) -> [float, tuple[int]]:
    keyword_to_log_prob_given_clue_funcs = (partial(clue_and_keyword_to_log_prob_func, clue) for clue in clues)
    log_expected_probability_given_clue_funcs = [partial(log_expectation, keyword_to_log_prob_given_clue_func) for keyword_to_log_prob_given_clue_func in keyword_to_log_prob_given_clue_funcs]

    # smells like dp, could see if caching helps if this ends up being bottleneck
    def max_expected_log_prob(var_indices=tuple(range(len(random_vars))), clue_indices=tuple(range(len(clues)))):
        if not clue_indices:
            return 0.0, tuple()
        
        clue_index, *remaining_clue_indices = clue_indices
        remaining_clue_indices = tuple(remaining_clue_indices)
                
        max_expected_log_probability = -math.inf
        best_guess = None

        # try each available random variable and see which yields best match probability
        for i, var_index in enumerate(var_indices):
            remaining_var_indices = var_indices[:i] + var_indices[i + 1:]
            
            max_subproblem_expected_log_probability, guess = max_expected_log_prob(remaining_var_indices, remaining_clue_indices)
            expected_log_probability = max_subproblem_expected_log_probability + log_expected_probability_given_clue_funcs[clue_index](random_vars[var_index])
            
            # update max log probability and corresponding code guess
            if expected_log_probability > max_expected_log_probability:
                max_expected_log_probability = expected_log_probability
                best_guess = (var_index,) + guess
                
        return max_expected_log_probability, best_guess
    
    return max_expected_log_prob()
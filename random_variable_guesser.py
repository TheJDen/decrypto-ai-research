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
    keyword_to_log_prob_given_clue_funcs = (partial(clue_and_keyword_to_log_prob_func, clue) for clue in clues)
    log_expected_probability_given_clue_funcs = [partial(log_expectation, keyword_to_log_prob_given_clue_func) for keyword_to_log_prob_given_clue_func in keyword_to_log_prob_given_clue_funcs]

    # smells like dp, could see if caching helps if this ends up being bottleneck
    def max_log_expected_prob_guess(var_indices=tuple(range(len(random_vars))), clue_index = 0) -> Guess:
        if clue_index == len(clues):
            return Guess() # probability 1, empty code
                        
        best_guess = Guess(-math.inf) # start at probability 0, is like starting max at -inf

        # try each available random variable and see which yields best match probability
        for i, var_index in enumerate(var_indices):
            remaining_var_indices = var_indices[:i] + var_indices[i + 1:]
            
            subproblem_best_guess = max_log_expected_prob_guess(remaining_var_indices, clue_index + 1)
            log_expected_probability = subproblem_best_guess.log_expected_probability + log_expected_probability_given_clue_funcs[clue_index](random_vars[var_index])
            
            # update max log probability and corresponding code guess
            if log_expected_probability > best_guess.log_expected_probability:
                best_guess = Guess(log_expected_probability, (var_index,) + subproblem_best_guess.code)
                
        return best_guess
    
    return max_log_expected_prob_guess()
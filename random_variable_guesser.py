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

    best_guess = [Guess(-math.inf)] * (len(clues) + 1)

    # when there are no clues, return an empty Guess with probability 1
    best_guess[0] = Guess()

    for i, clue in enumerate(clues):
        # curry clue into clue-and-keyword log probability strategy
        keyword_to_log_prob_given_clue_func = partial(clue_and_keyword_to_log_prob_func, clue)
        log_expected_probability_given_clue_func = partial(log_expectation, keyword_to_log_prob_given_clue_func)

        for var_index, random_var in enumerate(random_vars):
            log_expected_probability = best_guess[i].log_expected_probability + log_expected_probability_given_clue_func(random_var)
            # update best guess up to this clue
            if log_expected_probability > best_guess[i + 1].log_expected_probability:
                best_guess[i + 1] = Guess(log_expected_probability, best_guess[i].code + (var_index,))

    return best_guess[len(clues)]
import math
from functools import partial, reduce
from typing import Sequence

# abstract/encapsulate RadnomVariable so don't have to think about log logic as much

def softmax_combine(log_x, log_y):
    return log_x + math.log1p(math.exp(log_y - log_x))

class RandomVariable:
    def __init__(self, log_probabilities):
        self.log_probabilities = log_probabilities

    def log_expectation(self, key_to_log_val_func): # inject function, this is like log(E[f(X)])
        if not self.log_probabilities:
            return -math.inf
        
        return reduce(softmax_combine,
                      (log2_prob + key_to_log_val_func(key) for key, log2_prob in self.log_probabilities.items())
                      )

     
def max_expected_log_probability_guess(clue_and_keyword_to_log_prob_func, random_vars: Sequence[RandomVariable], clues: tuple[str]) -> [float, tuple[int]]:
    keyword_to_log_prob_given_clue = [partial(clue_and_keyword_to_log_prob_func, clue) for clue in clues]

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
            expected_log_probability = max_subproblem_expected_log_probability + random_vars[var_index].log_expectation(keyword_to_log_prob_given_clue[clue_index])
            
            # update max log probability and corresponding code guess
            if expected_log_probability > max_expected_log_probability:
                max_expected_log_probability = expected_log_probability
                best_guess = (var_index,) + guess
                
        return max_expected_log_probability, best_guess
    
    return max_expected_log_prob()

# naive log-probability strategy (too specific, but works to show concept)
def simple_log_prob_clue_and_keyword(clue, keyword):
    return (0.0 if clue == keyword else -math.inf)
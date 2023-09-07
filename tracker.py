from typing import Sequence
import math

class RandomVariableTracker:
    def __init__(self, random_vars: Sequence[dict], clue_probability_func, best_guess_func=max_log_probability_guess):
        self.random_vars = random_vars
        self.clue_probability_func = clue_probability_func
        self.best_guess_func = best_guess_func

    def clue_probability(self, clue: str, var_index: int) -> float:
        return self.clue_probability_func(self.random_vars, clue, var_index)
    
    def best_guess(self, clues: tuple[str]):
        return self.best_guess_func(self.clue_probability_func, self.random_vars, clues)

def max_log_probability_guess(clue_probability_func, random_vars: Sequence[dict], clues: tuple[str], var_indices=None) -> [float, tuple[int]]:
    # smells like dp, could see if caching helps 
    def max_log_prob(clues, var_indices):
        if not clues:
            return 0.0, tuple()
        
        clue, *remaining_clues = clues
        remaining_clues = tuple(remaining_clues)
                
        max_log_probability = -math.inf
        best_guess = None

        # try each available random variable and see which yields best match probability
        for i, var_index in enumerate(var_indices):
            remaining_var_indices = var_indices[:i] + var_indices[i + 1:]
            
            clue_probability = clue_probability_func(random_vars, clue, var_index)
            max_subproblem_log_probability, guess = max_log_prob(remaining_clues, remaining_var_indices)
            log_probability = max_subproblem_log_probability + (math.log(clue_probability) if clue_probability != 0 else -math.inf)
            
            # update max log probability and corresponding code guess
            if log_probability > max_log_probability:
                max_log_probability = log_probability
                best_guess = (var_index,) + guess
                
        return max_log_probability, best_guess
    
    var_indices = var_indices if var_indices is not None else tuple(range(len(random_vars))) 
    return max_log_prob(clues, var_indices)

def simple_clue_probability(random_vars: Sequence[dict], clue: str, var_index: int):
    return random_vars[var_index].get(clue, 0.0) # P(X_i = clue)


import math

class RandomVariableTracker:
    def __init__(self, random_vars: dict):
        self.random_vars = random_vars
        
    def clue_probability(self, clue: str, var_index: int):
        # this would need to be more refined for the general case
        return self.random_vars[var_index][clue] # P(X_i = clue)
        
    def max_log_probability_guess(self, clues: tuple[str], var_indices=None):
        if not clues:
            return 0.0, tuple()
        
        var_indices = var_indices if var_indices is not None else tuple(range(len(self.random_vars))) 
        clue, *remaining_clues = clues
        remaining_clues = tuple(remaining_clues)
                
        max_log_probability = -math.inf
        best_guess = None
        for i, var_index in enumerate(var_indices):
            remaining_var_indices = var_indices[:i] + var_indices[i + 1:]
            
            clue_probability = self.clue_probability(clue, var_index)
            max_subproblem_log_probability, guess = self.max_log_probability_guess(remaining_clues, remaining_var_indices)
            log_probability = max_subproblem_log_probability + (math.log(clue_probability) if clue_probability != 0 else -math.inf)
            
            if log_probability > max_log_probability:
                max_log_probability = log_probability
                best_guess = (var_index,) + guess
                
        return max_log_probability, best_guess
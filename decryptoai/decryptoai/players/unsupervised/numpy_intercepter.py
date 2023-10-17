import numpy as np
import numpy_guesser as npg
from functools import partial
# shape-preserving normalization

def normalized(log_probabilities: np.array) -> np.array:
    log_cumulative_density = npg.log_expected_probability(lambda _: np.NZERO, log_probabilities, np.zeros(len(log_probabilities)))
    log_normalization_factor = -log_cumulative_density # log(1) - ...
    return log_probabilities + log_normalization_factor


# initial distribution strategies

def equal_initial_distribution(word_index):
    log_probabilities = normalized(np.zeros(len(word_index)) - np.log(len(word_index)))
    keyword_indices = np.arange(len(word_index))
    return npg.NumpyRandomVariable(log_probabilities, keyword_indices)

def zipf_initial_distribution(word_index):
    log_probabilities = normalized(-np.log(np.arange(1, len(word_index) + 1))) # log(1) - ...
    keyword_indices = np.arange(len(word_index))
    return npg.NumpyRandomVariable(log_probabilities, keyword_indices)


# convenience initializer

def intercepter_random_variables(word_index, initial_distribution_func=equal_initial_distribution, num_vars=4):
    return [initial_distribution_func(word_index) for _ in range(num_vars)]


# optimistic update strat

def updated_random_vars(clue_and_keyword_to_log_probability_func, random_vars: list[npg.NumpyRandomVariable], clue_indices: np.ndarray, correct_code: np.ndarray, probability_reshape=lambda x: x):
    var_log_probabilities = np.array([random_var.log_probabilities for random_var in random_vars])
    var_keyword_indices = np.array([random_var.keyword_indices for random_var in random_vars])
    keyword_to_log_prob_vectorized = partial(clue_and_keyword_to_log_probability_func, clue_indices)
    log_association_probabilities = keyword_to_log_prob_vectorized(var_keyword_indices[correct_code])
    reshaped_log_associatipon_probabilities = probability_reshape(log_association_probabilities)
    var_log_probabilities[correct_code] += reshaped_log_associatipon_probabilities
    return [npg.NumpyRandomVariable(normalized(log_probabilities), keyword_indices) for log_probabilities, keyword_indices in zip(var_log_probabilities, var_keyword_indices)]

from dataclasses import dataclass
from functools import partial
from itertools import permutations
import numpy as np

ALL_CODES = np.array(list(permutations(range(4), 3)))
LOG_ZERO = np.float64(-100.0)

@dataclass
class NumpyRandomVariable:
    log_probabilities: np.array
    keyword_indices: np.array
  
    
def guesser_random_variables(keyword_card, word_index):
    return [NumpyRandomVariable(np.zeros(1), np.array([word_index[keyword]])) for keyword in keyword_card]

def np_clues(clues, word_index):
    return np.array([word_index[clue] for clue in clues])
    
# random variable size-reducing functions

def num_indices_for_cumulative_probability(log_probabilties: np.ndarray, keyword_indices_by_decreasing_probability: np.ndarray, cumulative_probability=1.0):
    # polling from a max heap until we reach cumulative_probability would be better for this theoretically
    # but lost NumPy speed may dominate
    probabilties_by_decreasing_probability = np.exp(log_probabilties[np.expand_dims(np.arange(len(log_probabilties)), axis=-1), keyword_indices_by_decreasing_probability])
    earliest_index = np.argmax(probabilties_by_decreasing_probability.cumsum(axis=-1) >= cumulative_probability, axis=-1)
    return earliest_index + 1


def random_vars_at_least_cumulative_probability(random_variables: list[NumpyRandomVariable], cumulative_probability=1.0):
    var_log_probabilities = np.array([random_variable.log_probabilities for random_variable in random_variables])
    keyword_indices_by_decreasing_probability = (-var_log_probabilities).argsort()
    num_indices = num_indices_for_cumulative_probability(var_log_probabilities, keyword_indices_by_decreasing_probability, cumulative_probability)
    reduced_keyword_indices = keyword_indices_by_decreasing_probability[:, slice(np.max(num_indices))]
    reduced_var_log_probabilities = var_log_probabilities[np.expand_dims(np.arange(len(var_log_probabilities)), axis=-1), reduced_keyword_indices]
    return [NumpyRandomVariable(log_probabilities, keyword_indices) for log_probabilities, keyword_indices in zip(reduced_var_log_probabilities, reduced_keyword_indices)]

# refactored guessing functions

def log_expected_probability(keyword_index_to_log_prob_func, log_probabilities: np.ndarray, keyword_indices: np.ndarray): # this is log equivalent of E[f(X)]
    # calculate terms of expectation sum definition
    log_terms = log_probabilities + keyword_index_to_log_prob_func(keyword_indices)
    # subtract max term to mitigate error
    # note: if we lose a lot of precision, we can omit the conversion and reduce, but it will be slower)
    max_log_term = np.max(log_terms, axis=-1)
    log_offset_terms = log_terms - np.expand_dims(max_log_term, axis=-1)
    # convert to regular probability world and evaluate sums to get expectation
    offset_expectation = np.sum(np.exp(log_offset_terms), axis=-1)
    # bring back to log world and add max term back
    log_expectation = np.log(offset_expectation) + max_log_term
    return log_expectation

def log_expected_probabilities_codes(clue_and_keyword_to_log_probability_func, random_variables: list[NumpyRandomVariable], clue_indices: np.ndarray, codes: np.ndarray = ALL_CODES):
    var_log_probabilities = np.array([random_variable.log_probabilities for random_variable in random_variables])
    var_keyword_indices = np.array([random_variable.keyword_indices for random_variable in random_variables])
    r_i, c_i = np.ogrid[slice(len(random_variables)), slice(len(clue_indices))]
    keyword_to_log_prob_vectorized = partial(clue_and_keyword_to_log_probability_func, clue_indices[c_i])
    log_expected_probabilities = log_expected_probability(keyword_to_log_prob_vectorized, var_log_probabilities[r_i], var_keyword_indices[r_i])
    return log_expected_probabilities[codes].trace(axis1=1, axis2=2)
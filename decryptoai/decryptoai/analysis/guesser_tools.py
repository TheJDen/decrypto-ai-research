from dataclasses import dataclass
import pandas
import numpy as np
import players.unsupervised.numpy_guesser as nguesser
import word2vec_loader.loader as wv_loader

@dataclass
class Suite:
    name: str
    clue_df: pandas.DataFrame
    correct_code_index: pandas.Series

@dataclass
class Strat:
    name: str
    strat_func: callable


def get_guess(word_index, strat_func, input_row):
    keyword_card = (input_row.keyword1, input_row.keyword2, input_row.keyword3, input_row.keyword4)
    clues = (input_row.clue1, input_row.clue2, input_row.clue3)
    wv_kw_card = map(wv_loader.official_keyword_to_word, keyword_card)
    random_vars = nguesser.guesser_random_variables(wv_kw_card, word_index)
    clue_indices = nguesser.np_clues(clues, word_index)
    code_log_probabilities = nguesser.log_expected_probabilities_codes(strat_func, random_vars, clue_indices)
    code_index_guess = np.argmax(code_log_probabilities)
    return pandas.Series([code_index_guess, code_log_probabilities[code_index_guess]], index=["code_index_guess", "log_expected_prob"])

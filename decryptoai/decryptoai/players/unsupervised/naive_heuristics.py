import numpy as np

def cosine_similarity(embedding, clue_index, keyword_index):
    clue_embedding = embedding[clue_index.squeeze()].transpose()
    keyword_embedding = embedding[keyword_index.squeeze()]
    return np.expand_dims(keyword_embedding.dot(clue_embedding), axis=-1)

# simple heuristics

def log_square_cosine_similarity(embedding, clue_index, keyword_index):
    similarity = cosine_similarity(embedding, clue_index, keyword_index)
    return 2 * np.log(np.abs(similarity))

def log_normalized_cosine_similarity(embedding, clue_index, keyword_index):
    similarity = cosine_similarity(embedding, clue_index, keyword_index)
    normalized_similiarity = (1 + similarity) / 2
    return np.log(normalized_similiarity)
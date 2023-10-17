import asyncio
import aiohttp
import json
import numpy as np
import random
from itertools import permutations

__author__ = "Jaden Rodriguez"


def datamuse_url(endpoint: str, words: list[str]): # can add stuff for prefix/suffix support later
    query_str = '+'.join(words)
    return f"https://api.datamuse.com/{endpoint}={query_str}"

async def fetch_text_response(session, url, return_id=None):
    # return ID let's us associate the result with a paramater
    # this allows us to know which word the reponse text is associated with
    # despite being called asynchronously
    async with session.get(url) as response:
        text = await response.text()
        return return_id, text

async def fetch_text_responses(urls, return_ids):
    async with aiohttp.ClientSession() as session:
        api_calls = [fetch_text_response(session, *args) for args in zip(urls, return_ids)]
        return [await response for response in asyncio.as_completed(api_calls)]

# process responses for local storage

def create_dataset_dict(responses):
    meaning_dataset = {}
    for word, response in responses:
        response_object = json.loads(response)
        if response_object:
            meaning_dataset[word] = response_object
    return meaning_dataset

async def load_dataset_from_path(path, endpoint: str, words):
    if not path.exists():
        if not path.parent.exists():
            path.parent.mkdir()
        urls = [datamuse_url(endpoint, [word]) for word in words]
        responses = await fetch_text_responses(urls, words)

        dataset = create_dataset_dict(responses)

        with open(str(path), 'w') as f:
            json.dump(dataset, f)
    else:
        with open(str(path)) as f:
            dataset = json.load(f)
    return dataset

def filter_illegal_cluewords(legal_clue_func, datamuse_dataset):
    filtered_dataset = {}
    for keyword, info in datamuse_dataset.items():
        legal_info = [word_info for word_info in info if legal_clue_func(keyword, word_info["word"])]
        filtered_dataset[keyword] = legal_info
    return filtered_dataset        

def clueword_from_dataset(datamuse_dataset, code_word, seed=400):
    candidate_words = []
    scores = []
    if code_word not in datamuse_dataset:
        return "garbage"
    for word_info in datamuse_dataset[code_word]:
        candidate_words.append(word_info["word"])
        scores.append(word_info["score"])
    np_scores = np.asarray(scores)
    probabilities = np_scores / np.sum(np_scores)
    [clue] = random.Random(seed).choices(candidate_words, probabilities)
    return clue

def clue_from_codewords(datamuse_dataset, codewords):
    return tuple(clueword_from_dataset(datamuse_dataset, word) for word in codewords)

def all_possible_codes(keyword_card_length=4, clue_length=3):
    return list(permutations(range(keyword_card_length), clue_length))

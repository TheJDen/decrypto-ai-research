import os
import gensim
import gensim.downloader
import gensim.models

GOOGLE_NEWS_PATH_NAME = "word2vec-google-news-300_c"

def load_word2vec_keyedvectors(path_str, limit=200_000):


    if not os.path.exists(path_str):
        google_news_wv = gensim.downloader.load("word2vec-google-news-300")
        google_news_wv.save_word2vec_format(path_str)
        del google_news_wv

    return gensim.models.KeyedVectors.load_word2vec_format(path_str, limit=limit)

def official_keyword_to_word(keyword: str):
    typos = { "CALENDA": "calendar"}
    if keyword in typos:
        return typos[keyword]
    proper_nouns = [
        "AFRICA",
        "CENTAUR",
        "CYCLOPS",
        "EGYPT",
        "FRANCE",
        "GERMANY",
        "PEGASUS",
        "QUEBEC",
        "RUSSIA"
    ]
    if keyword in proper_nouns:
        return keyword.capitalize()
    british = {
        "ARMOUR": "armor",
        "MOUSTACHE": "mustache",
        "THEATRE": "theater",
    }
    if keyword in british:
        return british[keyword]
    if '-' in keyword:
        if keyword == "SCIENCE-FICTION":
            keyword = keyword.replace('-', '_')
        keyword = keyword.replace('-', '')
    return keyword.lower()
import pathlib
import gensim
import gensim.downloader
import gensim.models
import decryptoai.config as cfg


__author__ = "Jaden Rodriguez"


def load_word2vec_keyedvectors(*, path: pathlib.Path = cfg.GOOGLE_NEWS_PATH_NAME, limit=200_000, debug=False):
    if not path.exists():
        if debug:
            print(f"{path.resolve()} not found, downloading")
        google_news_wv = gensim.downloader.load("word2vec-google-news-300")
        google_news_wv.save_word2vec_format(str(path))
        del google_news_wv

    return gensim.models.KeyedVectors.load_word2vec_format(str(path), limit=limit)

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
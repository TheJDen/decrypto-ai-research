from pathlib import Path

data = Path("data")
models = Path("models")

GOOGLE_NEWS_PATH_NAME = models / "word2vec-google-news-300_c"

MEANING_JSON_PATH = data / "meaning.json"
TRIGGERWORD_JSON_PATH = data / "trigger_word.json"

MEANING_CSV_PATH = data / "meaning_clues.csv"
TRIGGERWORD_CSV_PATH = data / "triggerword_clues.csv"
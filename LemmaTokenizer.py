from nltk import WordNetLemmatizer, word_tokenize, re
from sklearn.feature_extraction import stop_words


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        lemmas = list()
        for t in word_tokenize(articles):
            lemma = self.wnl.lemmatize(t)
            if lemma not in stop_words.ENGLISH_STOP_WORDS and re.search(r'^[a-zA-Z]+$', lemma) is not None:
                lemmas.append(lemma)
            lemmas.append(lemma)
        return lemmas

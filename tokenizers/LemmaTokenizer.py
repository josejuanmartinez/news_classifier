from nltk import WordNetLemmatizer, word_tokenize, re


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        result = list()
        for t in word_tokenize(articles):
            lemma = self.wnl.lemmatize(t)
            # if lemma not in stop_words.ENGLISH_STOP_WORDS and re.search(r'^[a-zA-Z]+$', lemma) is not None:
            # if lemma not in stop_words.ENGLISH_STOP_WORDS:
            result.append(lemma)
        return result




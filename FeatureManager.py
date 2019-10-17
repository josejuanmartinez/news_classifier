import pandas as pd

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from LemmaTokenizer import LemmaTokenizer


class FeatureManager:
    def __init__(self, file):
        self.clf = None
        self.data = pd.read_json(file)
        self.data = self.data.drop(['link'], axis=1)
        for col in self.data.columns:
            print(col)
        self.features = self.create_features()

    def stack_features(self, features):
        if len(features) < 1:
            return []
        X = features[0]
        for feature in features:
            X = sparse.hstack((X, feature))
        return X

    def print_features(self, feature):
        print(pd.DataFrame(feature.toarray(), columns=feature.get_feature_names()))

    def create_features(self):

        count_vect = CountVectorizer(tokenizer=LemmaTokenizer())
        tfidf_transformer = TfidfTransformer()

        author_bow_X_train_counts = count_vect.fit_transform(self.data['authors'])
        author_bow_X_train_tfidf = tfidf_transformer.fit_transform(author_bow_X_train_counts)

        count_vect = CountVectorizer(tokenizer=LemmaTokenizer())
        tfidf_transformer = TfidfTransformer()

        headline_bow_X_train_counts = count_vect.fit_transform(self.data['headline'])
        headline_bow_X_train_tfidf = tfidf_transformer.fit_transform(headline_bow_X_train_counts)

        count_vect = CountVectorizer(tokenizer=LemmaTokenizer())
        tfidf_transformer = TfidfTransformer()

        shortdescription_bow_X_train_counts = count_vect.fit_transform(self.data['short_description'])
        shortdescription_bow_X_train_tfidf = tfidf_transformer.fit_transform(shortdescription_bow_X_train_counts)

        return self.stack_features([author_bow_X_train_tfidf, headline_bow_X_train_tfidf, shortdescription_bow_X_train_tfidf ])

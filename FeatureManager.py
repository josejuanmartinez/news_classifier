import pandas as pd
from nltk import word_tokenize

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from Cleanser import Cleanser
from tokenizers.LemmaTokenizer import LemmaTokenizer


class FeatureManager:
    def __init__(self, data, combination, logger, cleanse, embeddings):
        self.logger = logger
        self.logger.info("PREPARING FEATURES {}".format(combination))
        self.logger.info("***********************")
        self.clf = None
        self.data = data

        if cleanse:
            self.data = Cleanser(self.data).data

        self.features = self.create_features(self.data, combination, embeddings)

    def stack_features(self, features):
        if len(features) < 1:
            raise ValueError("No features found")

        X = features[0]
        if len(features) == 1:
            return X

        for feature in features:
            X = sparse.hstack((X, feature))
        return X

    def print_features(self, feature):
        print(pd.DataFrame(feature.toarray(), columns=feature.get_feature_names()))

    def create_features(self, data, combination, embeddings):
        features_tfidf = []
        for feature in combination:
            self.logger.info("Processing feature {}".format(feature))

            if feature == 'd2v':
                if embeddings is not None:
                    features_tfidf.append(embeddings)
                else:
                    continue
            else:

                count_vect = CountVectorizer(tokenizer=LemmaTokenizer())
                tfidf_transformer = TfidfTransformer()

                X_train_counts = count_vect.fit_transform(data[feature])
                X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
                features_tfidf.append(X_train_tfidf)

        return self.stack_features(features_tfidf)

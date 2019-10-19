import argparse
import os
import sys

import pandas as pd

from nltk import word_tokenize
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from embeddings.embedding_vectorizer import EmbeddingVectorizer
from loggers.ClassifierLogger import ClassifierLogger
from FeatureManager import FeatureManager
from itertools import chain, combinations

FEATURES = ['d2v', 'link', 'authors', 'headline', 'short_description']
CLEANSE = True
MIN_COMBINATION_SIZE = len(FEATURES)
EMBEDDINGS_CORPUS = None
ALGORITHMS = ["Support Vector Machine"]
TEST_SIZE = 0.2
# ["Naive Bayes", "Decision Tree", "Adaboost", "Support Vector Machine", "Random Forest", "Gradient Descent"]


class Classifier:
    def __init__(self, feature_manager, algo_list, logger):
        self.logger = logger
        self.data = feature_manager.data
        self.feature_manager = feature_manager
        self.algo_list = algo_list

    @staticmethod
    def get_classifier(algo):
        if algo == "Gradient Boost":
            return GradientBoostingClassifier()
        elif algo == "Random Forest":
            return RandomForestClassifier()
        elif algo == "Adaboost":
            return AdaBoostClassifier()
        elif algo == "Decision Tree":
            return  DecisionTreeClassifier()
        elif algo == "Naive Bayes":
            return  BernoulliNB()
        elif algo == "Gradient Descent":
            return  SGDClassifier()
        elif algo == "Support Vector Machine":
            return LinearSVC()
        elif algo == "MLPC":
            # NEURAL
            return MLPClassifier(activation='logistic',  batch_size='auto',
            early_stopping=True, hidden_layer_sizes=(100,), learning_rate='adaptive',
            learning_rate_init=0.1, max_iter=5000, random_state=1,
            solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
            warm_start=False)
        return 0

    def train(self):
        # Train the model

        X = self.feature_manager.features
        y_1 = self.data['category'].values.tolist()

        # Split dataset into training set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_1, test_size=TEST_SIZE,
                                                            random_state=2)

        # Instantiate Classifying Algorithm
        for algo in self.algo_list:
            self.clf = self.get_classifier(algo)
            self.logger.info(algo)
            self.logger.info("====")
            self.logger.info("Training...")
            clf = self.clf.fit(self.X_train, self.y_train)
            self.logger.info("Trained!")
            self.logger.info("Evaluating...")
            self.evaluate(clf)
            self.logger.info("Evaluated!")

    def evaluate(self, clf):
        # Predict and evaluate the response for test dataset
        y_pred = clf.predict(self.X_test)
        self.logger.info("-> Predicted: {}".format(y_pred))
        self.logger.info("-> Accuracy: {}".format(accuracy_score(self.y_test, y_pred)))


if __name__ == "__main__":
    ''' Definition of command line arguments for ArgumentParser '''
    parser = argparse.ArgumentParser(description='Runs Classifier')
    parser.add_argument('--cleanse', action='store_true', dest='cleanse', default=False)
    parser.add_argument('--features', action='store', nargs='+', dest='features', default=FEATURES)
    parser.add_argument('--algo', action='store', nargs='+', dest='algo', default=ALGORITHMS)
    parser.add_argument('--min_feat_size', action='store', dest='min_comb_size', type=int, default=MIN_COMBINATION_SIZE)
    parser.add_argument('--embeddings', action='store', dest='embeddings', default=None)
    parser.add_argument('--test_size', action='store', type=float, dest='test_size', default=TEST_SIZE)

    ''' Parsing of arguments from command line'''
    args = parser.parse_args(sys.argv[1:])

    ''' Configuration of parameters to be overwriten '''
    if args.cleanse:
        CLEANSE = args.cleanse

    if args.features:
        FEATURES = args.features

    if args.algo:
        ALGORITHMS = args.algo

    if args.min_comb_size:
        MIN_COMBINATION_SIZE = args.min_comb_size

    if args.embeddings:
        EMBEDDINGS_CORPUS = args.embeddings

    if args.test_size:
        TEST_SIZE = args.test_size

    # Instantiate Logger class
    logger = ClassifierLogger().get_logger()

    # Get the corpus file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    corpus = os.path.join(dir_path, "resources", "News_category_train.json")
    data = pd.read_json(corpus)

    # I get combinations of features to see if they are relevant
    combinations = chain(*map(lambda x: combinations(FEATURES, x), range(0, len(FEATURES) + 1)))

    embeddings = None
    if 'd2v' in FEATURES and EMBEDDINGS_CORPUS is not None:
        # Get the embeddings
        logger.info("Loading embeddings...")
        model = EmbeddingVectorizer(os.path.join(dir_path, 'embeddings', EMBEDDINGS_CORPUS, 'doc2vec.bin'))

        logger.info("Creating d2v feature sparse matrix")
        rows = []
        for idx, element in enumerate(data['headline']):
            v_string = [t for t in word_tokenize(element)]
            vec = model.infer_vector(v_string, 0.01, 1000)
            rows.append(vec)
            if idx % 1000 == 0:
                logger.info("Processed {} rows".format(idx))
        logger.info("Finished. Rows: {}".format(len(rows)))
        embeddings = sparse.csr_matrix(rows)

    # Train and evaluate all combinations

    logger.info("Cleasing active? {}".format(CLEANSE))
    logger.info("Embeddings active? {}".format('d2v' in FEATURES and EMBEDDINGS_CORPUS is not None))

    for combination in combinations:
        if len(combination) < MIN_COMBINATION_SIZE:
            continue
        feat_manager = FeatureManager(data, combination, logger, CLEANSE, embeddings)

        # Best models are SVM and SGD
        classifierNB = Classifier(feat_manager, ALGORITHMS, logger)

        classifierNB.train()




import os
import sys

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from ClassifierLogger import ClassifierLogger
from FeatureManager import FeatureManager
from itertools import chain, combinations

class Classifier:
    def __init__(self, feature_manager, algo_list, logger):
        self.logger = logger
        self.data = feature_manager.data
        self.feature_manager = feature_manager
        self.algo_list = algo_list
        
    def get_classifier(self, algo):
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
            return MLPClassifier(activation='logistic',  batch_size='auto',
            early_stopping=True, hidden_layer_sizes=(100,), learning_rate='adaptive',
            learning_rate_init=0.1, max_iter=5000, random_state=1,
            solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
            warm_start=False)
        return 0

    def train(self, cleanse):
        X = self.feature_manager.cleansed_features if cleanse else self.feature_manager.features
        y_1 = self.data['category'].values.tolist()

        # Split dataset into training set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_1, test_size=0.3,
                                                            random_state=1)  # 70% training and 30% test

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
        # Predict the response for test dataset
        y_pred = clf.predict(self.X_test)
        self.logger.info("-> Predicted: {}".format(y_pred))
        self.logger.info("-> Correct: {}".format(self.y_test))
        self.logger.info("-> Accuracy: {}".format(accuracy_score(self.y_test, y_pred)))


if __name__ == "__main__":
    logger = ClassifierLogger().get_logger()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    corpus = os.path.join(dir_path, "resources", "News_category_train.json")

    features = ['authors', 'headline', 'short_description']
    combinations = chain(*map(lambda x: combinations(features, x), range(0, len(features) + 1)))

    for combination in combinations:
        if len(combination) == 0:
            continue
        feat_manager = FeatureManager(corpus, combination, logger)
        # classifierNB = Classifier(feat_manager, ["Naive Bayes", "Decision Tree", "Adaboost", "Support Vector Machine", "Random Forest", "Gradient Descent"])
        classifierNB = Classifier(feat_manager, ["Support Vector Machine", "Gradient Descent"] ,logger)
        # self.logger.info("TRAINING WITHOUT CLEANSING")
        # classifierNB.train(cleanse=False)
        classifierNB.train(cleanse=True)




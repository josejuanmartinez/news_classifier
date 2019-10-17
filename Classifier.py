import os

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from FeatureManager import FeatureManager


class Classifier:

    def __init__(self, feat_manager, algo="NB"):
        self.feat_manager = feat_manager
        self.algo = algo

    def get_classifier(self):
        if self.algo == "GBT":
            return GradientBoostingClassifier()
        elif self.algo == "RF":
            return  RandomForestClassifier()
        elif self.algo == "ADB":
            return AdaBoostClassifier()
        elif self.algo == "DT":
            return  DecisionTreeClassifier()
        elif self.algo == "NB":
            return  BernoulliNB()
        elif self.algo == "SGD":
            return  SGDClassifier()
        elif self.algo == "SVC":
            return LinearSVC()
        elif self.algo == "MLPC":
            return MLPClassifier(activation='logistic',  batch_size='auto',
            early_stopping=True, hidden_layer_sizes=(100,), learning_rate='adaptive',
            learning_rate_init=0.1, max_iter=5000, random_state=1,
            solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True,
            warm_start=False)
        return 0

    def train(self):
        X = self.feat_manager.features
        y_1 = self.feat_manager.data['category'].values.tolist()

        # Split dataset into training set and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_1, test_size=0.3,
                                                            random_state=1)  # 70% training and 30% test

        # Instantiate Classifying Algorithm
        self.clf = self.get_classifier()

        print("Training...")
        self.clf = self.clf.fit(self.X_train, self.y_train)
        print("Trained!")

    def evaluate(self):
        # Predict the response for test dataset
        y_pred = self.clf.predict(self.X_test)
        print("Evaluating...")
        print("-> Predicted: {}".format(y_pred))
        print("-> Correct: {}".format(self.y_test))
        print("-> Accuracy: {}".format(accuracy_score(self.y_test, y_pred)))
        print("Evaluated!")


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    corpus = os.path.join(dir_path, "resources", "News_category_train.json")
    feat_manager = FeatureManager(corpus)
    print("NAIVE BAYES")
    print("===========")
    classifierNB = Classifier(feat_manager, "NB")
    classifierNB.train()
    classifierNB.evaluate()
    print("")
    print("DECISION TREES")
    print("===========")
    classifierDT = Classifier(feat_manager, "DT")
    classifierDT.train()
    classifierDT.evaluate()
    print("")
    print("ADABOOST")
    print("===========")
    classifierADB = Classifier(feat_manager, "ADB")
    classifierDT.train()
    classifierDT.evaluate()
    print("")
    print("SUPPORT VECTOR MACHINE")
    print("===========")
    classifierRF = Classifier(feat_manager, "SVC")
    classifierRF.train()
    classifierRF.evaluate()
    print("")
    print("RANDOM FORESTS")
    print("===========")
    classifierRF = Classifier(feat_manager, "RF")
    classifierRF.train()
    classifierRF.evaluate()
    print("")
    print("STOCHASTIC GRADIENT DESCENT")
    print("===========")
    classifierRF = Classifier(feat_manager, "SGD")
    classifierRF.train()
    classifierRF.evaluate()
    """print("")
    print("GRADIENT BOOST")
    print("===========")
    classifierGBT = Classifier(feat_manager, "GBT")
    classifierGBT.train()
    classifierGBT.evaluate()
    """




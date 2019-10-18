import logging
import sys


class ClassifierLogger:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        file_handler = logging.FileHandler(filename='Classifier.log')
        #stdout_handler = logging.StreamHandler(sys.stdout)

        self.logger = logging.getLogger()

        #self.logger.addHandler(stdout_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

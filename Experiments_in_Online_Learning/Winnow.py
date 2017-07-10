__author__ = 'shail_000'

import settings
import math


# describes the class for Winnow  algorithm modified for calculating curves experiment
class Winnow:
    def __init__(self, n, no_of_examples, weight_vector, learning_rate,
                 estimated_margin, threshold, training_file_path=None, testing_file_path=None):
        self.n = n
        self.no_of_examples = no_of_examples
        self.weight_vector = weight_vector
        self.learning_rate = learning_rate
        self.estimated_margin = estimated_margin
        self.threshold = threshold
        self.training_file_name = training_file_path
        if testing_file_path is None:
            self.testing_file_name = ""
        else:
            self.testing_file_name = testing_file_path
        self.list_of_training_objects = settings.create_list_of_training_objects(training_file_path)
        self.mistakes = 0
        self.accuracy = 0.0
        self.mistakes_list = []

    # takes the dot product of weights and feature vector of a given example
    # only those indices of weight vector are updated for which index the value of x = 1
    def calculate_activation(self, feature_vector):
        dot_product_result = 0.0
        for feature in feature_vector:
            feature_weight = self.weight_vector[feature]
            dot_product_result += feature_weight
        return dot_product_result - self.threshold

    # updates weight vector
    def update_weight_vector(self, feature_vector, y):
        for feature in feature_vector:
            self.weight_vector[feature] *= math.pow(self.learning_rate, y)

    # learns parameters from the training data and calculates cumulative mistakes at every 1000 instances
    def learn(self):
        for t in range(0, settings.T):
            for x in range(0, self.no_of_examples):
                if x % 1000 == 0:
                    self.mistakes_list.append(self.mistakes)
                example = self.list_of_training_objects[x]
                activation = self.calculate_activation(example.feature_vector)
                if example.label * activation <= 0:
                    self.mistakes += 1
                if example.label * activation <= self.estimated_margin:
                    self.update_weight_vector(example.feature_vector, example.label)

    # predicts the label and updates accuracy
    def decide(self):
        errors_made = 0
        test_file_list = settings.read_file_into_list(self.testing_file_name)
        for example in test_file_list:
                actual_label = example[0]
                activation = self.calculate_activation(example[1:len(example)])
                predicted_label = settings.predict_label(activation, 0)
                if actual_label != predicted_label:
                    errors_made += 1
        self.accuracy = ((self.no_of_examples-errors_made)/self.no_of_examples) * 100



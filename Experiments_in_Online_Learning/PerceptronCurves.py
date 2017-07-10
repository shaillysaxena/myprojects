__author__ = 'shail_000'

import settings


# describes the class for Perceptron algorithm modified for calculating curves experiment
class Perceptron:
    def __init__(self, n, no_of_examples, weight_vector, bias, learning_rate,
                 estimated_margin, training_file_path, testing_file_path):
        self.n = n
        self.no_of_examples = no_of_examples
        self.weight_vector = weight_vector
        self.bias = bias
        self.learning_rate = learning_rate
        self.estimated_margin = estimated_margin
        self.training_file_name = training_file_path
        self.testing_file_name = testing_file_path
        self.list_of_training_objects = settings.create_list_of_training_objects(training_file_path)
        self.mistakes = 0
        self.accuracy = 0.0
        self.mistakes_list = []
        self.consecutive_no_mistakes = 0

    # takes the dot product of weights and feature vector of a given example
    # only those indices of weight vector are updated for which index the value of x = 1
    def calculate_activation(self, feature_vector):
        dot_product_result = 0.0
        for feature in feature_vector:
            feature_weight = self.weight_vector[feature]
            dot_product_result += feature_weight
        return dot_product_result + self.bias

    # updates the values of weight vector
    def update_weight_vector(self, feature_vector, y):
        for feature in feature_vector:
            self.weight_vector[feature] += (y * self.learning_rate)

    # updates the bias
    def update_bias(self, y):
        self.bias += float(y * self.learning_rate)

    # learns parameters from the training data and breaks when S consecutive examples do not update the weight vector
    def learn(self):
        S = 35
        for x in range(0, self.no_of_examples):
            example = self.list_of_training_objects[x]
            activation = self.calculate_activation(example.feature_vector)
            if example.label * activation <= 0:
                self.mistakes += 1
                self.consecutive_no_mistakes = 0
            else:
                self.consecutive_no_mistakes += 1
            if self.consecutive_no_mistakes == S:
                return 0

            if example.label * activation <= self.estimated_margin:
                self.update_weight_vector(example.feature_vector, example.label)
                self.update_bias(example.label)
        return -1

    # predicts the label and updates accuracy
    def decide(self):
        test_file_list = settings.read_file_into_list(self.testing_file_name)
        errors_made = 0
        for example in test_file_list:
                actual_label = example[0]
                activation = self.calculate_activation(example[1:len(example)])
                predicted_label = settings.predict_label(activation, 0)
                if actual_label != predicted_label:
                    errors_made += 1
        self.accuracy = ((self.no_of_examples-errors_made)/self.no_of_examples) * 100

__author__ = 'shail_000'

# General Idea: the feature vectors and weight vector should be, ideally, of the same
# size i.e. the size of the number of words in the vocabulary file.
# But we will be wasting a lot of space for the feature vectors since, every example is small
# subset of the vocabulary.
# Therefore, to save memory and hence, increasing computation speed, here, the size of a
# feature vector is equal to the number of words in the given example
# There is a one-on-one mapping between (index - 1) of feature vector of training example
# and the same index in the weight_list
# The size of the weight_list is equal to the size of the number of words in the vocabulary file

import TrainingExample
import random
import math

vocab_path = "libsvm/features.lexicon"
prediction_file_path = "output/predictions.lr.txt"
labels_file_path = "output/labels.txt"

T = 10   # defining the number of rounds to be considered
learning_rate = 0.5
threshold_value = 0.5


# opens a file and reads it into a list, ...
def read_file_into_list(file_path):
    with open(file_path) as given_file:
        examples = given_file.readlines()
    return examples


# takes a string and splits it on the given splitter
def splitting(a_list, splitter):
    return a_list.split(splitter)


# function that counts the number of lines a file has
# takes in the path of a file
# here, takes the path of vocabulary file as input
# used to find the total number of elements in the weights list
def get_no_of_lines(vocab_file_path):
    with open(vocab_file_path) as vocab_file:
        vocab_contents_list = vocab_file.readlines()
        return len(vocab_contents_list)


# initializes the weight_list to 0
def create_initial_weight_list(list_length):
    weight_list = [0.0] * list_length
    return weight_list


# writes the given label to the file at file_path
def write_to_file(label, file_path):
    with open(file_path, 'a') as f:
        f.write(str(label))
        f.write("\n")
        f.close()


# an example is separated by a newline in the training file
# an example is a list of strings
# this list when separated by a spaces, gives a list of a feature and its weight
# every feature has weight attached to it separated by a colon
# weights are ignored and feature vectors are converted to zero-based indices
# by subtracting one from the feature read from the file
# this is done so that the feature vectors have one-to-one relationship with the
# weight_list indices
def create_list_of_training_objects(training_path):
    list_of_training_objects = []
    training_examples = read_file_into_list(training_path)
    for every_example in training_examples:
        list_of_weighted_features = splitting(every_example, " ")
        label = list_of_weighted_features[0]
        te = TrainingExample.TrainingExample(label)
        for i in range(1, len(list_of_weighted_features)):
            (feature, weight) = splitting(list_of_weighted_features[i], ":")
            te.add_feature_to_feature_list(int(feature) - 1)
        list_of_training_objects.append(te)
    return list_of_training_objects


# class to implement the logistic regression algorithm using Stochastic Gradient Descent
# contains:
# the number of words in the vocabulary file as vocab_word_count
# the list that contains the weight vector
# a list containing objects corresponding to every training example
# the bias - w0
class LogisticRegression:
    def __init__(self, training_file_path):
        self.vocab_list_size = get_no_of_lines(vocab_path)
        self.weight_list = create_initial_weight_list(self.vocab_list_size)
        self.list_of_training_objects = create_list_of_training_objects(training_file_path)
        self.w0 = 0.0

    # takes the dot product of weights and feature vector of a given example
    # only those indices of weight vector are updated for which index the value of x = 1
    def calculate_w_transpose_x(self, feature_vector):
        dot_product_result = 0.0
        for every_feature in feature_vector:
            weight_for_this_feature = self.weight_list[every_feature]
            dot_product_result += weight_for_this_feature
        return dot_product_result

    # gives the value of the sigmoid function for a training example
    def calculate_sigma(self, feature_vector):
        w_t_x = self.calculate_w_transpose_x(feature_vector)
        w = w_t_x + self.w0
        return 1.0/(1.0 + math.exp(- w))

    # updates the value of weight vector depending on the value of delta
    # updates are made only for those indices of weight for which index the x in feature vector is 1
    def update_weight_list(self, delta, feature_vector):
        product = float(learning_rate * delta)
        for every_feature in feature_vector:
            self.weight_list[every_feature] += product

    # learn the given training data by logistic regression using SGD
    def learn(self):
        print("Learning")
        no_of_training_examples = len(self.list_of_training_objects)

        for t in range(0, T):
            for x in range(0, no_of_training_examples):
                random_index = random.randint(0, no_of_training_examples-1)
                random_example = self.list_of_training_objects[random_index]
                sigma = self.calculate_sigma(random_example.feature_vector)
                delta = random_example.label_numerical_value - sigma

                self.update_weight_list(delta, random_example.feature_vector)

                self.w0 += (learning_rate * delta)

    # makes predictions on the given testing data
    def decide(self, testing_file_path):
        print("Deciding")
        with open(testing_file_path) as test_file:
            test_file_list = test_file.readlines()
            for every_example in test_file_list:
                feature_list = []
                list_of_weighted_features = splitting(every_example, " ")
                actual_label = list_of_weighted_features[0]
                write_to_file(actual_label, labels_file_path)
                for i in range(1, len(list_of_weighted_features)):
                    (feature, weight) = splitting(list_of_weighted_features[i], ":")
                    feature_list.append(int(feature) - 1)
                y_prime = self.calculate_sigma(feature_list)
                if y_prime >= threshold_value:
                    predicted_label = 6.0
                else:
                    predicted_label = 2.0

                write_to_file(predicted_label, prediction_file_path)





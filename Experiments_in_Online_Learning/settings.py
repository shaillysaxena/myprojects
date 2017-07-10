__author__ = 'shail_000'
import TrainingExample
import os

# global variables
training_config1 = "Training Files/Tuning/Online.config1.training"
training_config2 = "Training Files/Tuning/Online.config2.training"
config1_D1 = "Training Files/Tuning/D1.config1.training"
config1_D2 = "Training Files/Tuning/D2.config1.training"
config2_D1 = "Training Files/Tuning/D1.config2.training"
config2_D2 = "Training Files/Tuning/D2.config2.training"
n40 = "Training Files/Curves/Train/l_10_m_20_n_40.txt"
n80 = "Training Files/Curves/Train/l_10_m_20_n_80.txt"
n120 = "Training Files/Curves/Train/l_10_m_20_n_120.txt"
n160 = "Training Files/Curves/Train/l_10_m_20_n_160.txt"
n200 = "Training Files/Curves/Train/l_10_m_20_n_200.txt"
n40_test = "Training Files/Curves/Test/n40_test.txt"
n80_test = "Training Files/Curves/Test/n80_test.txt"
n120_test = "Training Files/Curves/Test/n120_test.txt"
n160_test = "Training Files/Curves/Test/n160_test.txt"
n200_test = "Training Files/Curves/Test/n200_test.txt"
m100 = "Training Files/Batch/Train/m100.txt"
m500 = "Training Files/Batch/Train/m500.txt"
m1000 = "Training Files/Batch/Train/m1000.txt"
m100_test = "Training Files/Batch/Test/m100_test.txt"
m500_test = "Training Files/Batch/Test/m500_test.txt"
m1000_test = "Training Files/Batch/Test/m1000_test.txt"
D1m100 = "Training Files/Batch/D/D1m100.txt"
D1m500 = "Training Files/Batch/D/D1m500.txt"
D1m1000 = "Training Files/Batch/D/D1m1000.txt"
D2m100 = "Training Files/Batch/D/D2m100.txt"
D2m500 = "Training Files/Batch/D/D1m500.txt"
D2m1000 = "Training Files/Batch/D/D1m1000.txt"

T = 20
n1 = 500
n2 = 1000
m = 5000
config_instances = 50000


# opens a file and reads it into a list, ...
def read_file_into_list(file_path):
    feature_list = []
    if os.path.isfile("temp.txt"):
        os.remove("temp.txt")
    with open(file_path) as given_file:
        unprocessed_examples = given_file.readlines()
        for example in unprocessed_examples:
            example_feature_list = []
            list_of_weighted_features = example.split(" ")
            label = list_of_weighted_features[0]
            example_feature_list.append(label)
            example_len = len(list_of_weighted_features)
            for i in range(1, example_len):
                (feature, weight) = list_of_weighted_features[i].split(":")
                example_feature_list.append(int(feature) - 1)
            feature_list.append(example_feature_list)
    return feature_list


# initializes the weight_list to 0
def create_initial_weight_list(list_length):
    weight_list = [0.0] * list_length
    return weight_list


# initializes the weight_list to 0
def create_initial_weight_list1(list_length):
    weight_list = [1.0] * list_length
    return weight_list


# writes the given label to the file at file_path
def write_label_to_file(label, file_path):
    with open(file_path, 'a') as f:
        f.write(str(label))
        f.write("\n")
        f.close()


# writes to a file
def write_to_file(string, filename):
    with open(filename, 'a') as temp:
        temp.write(string)
        temp.write("\n")
        temp.close()


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
    feature_list = read_file_into_list(training_path)
    for example in feature_list:
        te = TrainingExample.TrainingExample(example[0], example[1:len(example)])
        list_of_training_objects.append(te)
    return list_of_training_objects


# predicts a label after comparing it with the threshold
def predict_label(activation, threshold):
    if activation >= threshold:
        return "+1"
    else:
        return "-1"


# creates a list with a gap of 1000 between values
def create_iter_list():
    x = 0
    a_list = []
    while x != 50000:
        a_list.append(x)
        x += 1000
    return a_list

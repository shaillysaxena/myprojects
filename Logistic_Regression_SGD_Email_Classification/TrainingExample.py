__author__ = 'shail_000'


# maps the label value of 2.0 to 0.0 and 6.0 to 1.0
def set_label(label):
    if label == "2.0":
        return 0.0
    else:
        return 1.0


# class that defines a training example:
# contains:
# label - of the actual label of the given training example
# label_numerical_value - mapped value of label to y=(0,1)
# feature_vector - list of indices for which x is present; the
# ones that are not present are not stored
# this is done for better memory management; every feature vector is
# of the size of the number of words given for it in the training example
class TrainingExample:
    def __init__(self, label):
        self.label = label
        self.label_numerical_value = set_label(label)
        self.feature_vector = []

    # adds a feature to the feature list
    def add_feature_to_feature_list(self, feature):
        self.feature_vector.append(feature)

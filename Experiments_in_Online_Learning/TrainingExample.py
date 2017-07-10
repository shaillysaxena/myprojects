__author__ = 'shail_000'


# a training example consisting of a the label and feature vector
class TrainingExample:
    def __init__(self, label, feature_vector):
        self.label = int(label)
        self.feature_vector = feature_vector


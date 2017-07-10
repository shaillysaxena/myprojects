__author__ = 'shail_000'


# a LabelType is a type of email folder
# contains:
# name of the folder
# occurrences of this folder in the entire training set
# prior value of the folder
# a dictionary containing all words and their counts
# posterior of every word w.r.t. to the folder
# numerical value which is a mapping to the name of the folder
# count of words in the emails of that folder
class LabelType:
    def __init__(self, name, count, prior, dictionary, num_value, word_count):
        self.name = name
        self.count = count
        self.prior = prior
        self.dict = dictionary
        self.conditional_p = dict()
        self.num_value = num_value
        self.word_count = word_count

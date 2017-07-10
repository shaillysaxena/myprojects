__author__ = 'shail_000'

import LogisticRegression as lReg
import os

training_file_path = "libsvm/train.libsvm"
testing_file_path = "libsvm/test.libsvm"


try:
    os.remove("output/predictions.lr.txt")
    os.remove("output/labels.txt")

except IOError:
    pass


def main():
    lr = lReg.LogisticRegression(training_file_path)
    lr.learn()
    lr.decide(testing_file_path)
    print("Predictions made! Check predictions.lr.txt in output folder")

if __name__ == '__main__':
    main()
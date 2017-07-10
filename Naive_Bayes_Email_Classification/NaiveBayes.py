__author__ = 'shail_000'
import label_type as l
import math

vocab_filename = "processed/vocabulary.txt"
test_file_path = "processed/test.txt"
prediction_file_path = "output/predictions.nb.txt"
test_file_list = []

# opens a file and returns the dictionary form of it
# key is the word and value is count
def create_dict(filename):
    try:
        train_file = open(filename)
        created_dict = dict()
        for every_line in train_file:
            try:
                a_line = every_line.split("\n")
                (word, count) = a_line[0].split(" ")
                created_dict[word] = int(count)

            except ValueError:
                pass
        train_file.close()
        return created_dict
    except IOError:
        print(filename + " not found")


# counts the total number of lines in a given file
def calculate_no_of_lines(file_path):
    try:
        vocab_file = open(file_path)
        vocab_count = sum(1 for line in vocab_file)
        vocab_file.close()
        return vocab_count
    except IOError:
        print("error opening vocab file")


# writes the predicted label to the output file
def write_to_file(predicted_label):
    try:
        with open(prediction_file_path, 'a') as f:
            f.write(predicted_label)
            f.write("\n")
        f.close()

    except IOError:
        print("Cannot open testing file for predictions")


# class NaiveBayes
# contains:
# the dictionary of the entire vocabulary
# label_mapping is a dictionary of email-folder objects as keys and email-folder names as values
# y_primes as a dictionary where key is the email-folder name and values is the value of y-prime for that folder
class NaiveBayes:
    def __init__(self):
        self.vocabulary = create_dict(vocab_filename)
        self.label_mapping = dict()
        self.vocab_count = calculate_no_of_lines(vocab_filename)
        self.y_primes = dict()

    # def learn
    def learn(self):
        print("Learning...")
        labels_list = self.label_mapping.values()
        vocab_keys_list = self.vocabulary.keys()    # entire vocab
        for current_label in labels_list:   # current_label is an object
            for current_key in vocab_keys_list:   # current_key is a word in vocabulary
                word_count = 0
                if current_key in current_label.dict:
                    word_count = current_label.dict.get(current_key)
                current_label.conditional_p[current_key] = (word_count + 1) / (self.vocab_count + current_label.word_count)

    # def decide
    def decide(self):
        print("Predicting labels...")
        try:
            # opening the test file and storing all emails in a list; a list-of-lists format
            test_file_contents = open(test_file_path)
            for each_line in test_file_contents:
                one_test_mail = each_line.split("\n")
                test_file_list.append(one_test_mail)
            test_file_contents.close()
            labels_list = self.label_mapping.values()
            # decision-making begins here
            for current_email in test_file_list:    # a list containing a string
                # separate all the words from a test email
                words_in_email = current_email[0].split(" ")
                # for every email-folder type:
                for current_label in labels_list:   # current_label is an object
                    sum_of_logs = 0.0
                    # for every word in the test email, ignoring the first word which is the actual label
                    for current_index in range(1, len(words_in_email)):
                        # if word in test email is present in the  dictionary of current email-folder
                        if words_in_email[current_index] in current_label.dict:
                            # pick that posterior from email-folder and add to sum of logs
                            sum_of_logs += math.log(current_label.conditional_p.get(words_in_email[current_index]))
                            # if a new word is found, add the new posterior with |V| + 1
                        else:
                            sum_of_logs += math.log(1 / (self.vocab_count + 1 + current_label.word_count))
                    # store the y-prime value for this email-folder type
                    self.y_primes[current_label] = math.log(current_label.prior) + sum_of_logs
                # find the max of these primes
                stats = self.y_primes
                predicted_label = max(stats.keys(), key=(lambda key: stats[key])).num_value
                # write the numerical value of the predicted email-folder to the predictions file
                write_to_file(predicted_label)

        except IOError:
            print("Cannot open test file")

    # adds a label to the NaiveBayes algorithm
    def add_label(self, name, number, count, path):
        dictionary = create_dict(path)
        word_count = sum(dictionary.values())
        prior = count/self.vocab_count
        new_label = l.LabelType(name, count, prior, dictionary, number, word_count)
        self.label_mapping[name] = new_label

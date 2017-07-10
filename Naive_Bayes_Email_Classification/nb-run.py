__author__ = 'shail_000'

import NaiveBayes
import os

try:
    os.remove("output/prediction.nb.txt")

except IOError:
    pass

articles_path = "processed/articles.train.txt"
corporate_path = "processed/corporate.train.txt"
enron_t_s_path = "processed/enron_t_s.train.txt"
enron_travel_path = "processed/enron_travel_club.train.txt"
hea_nesa_path = "processed/hea_nesa.train.txt"
personal_path = "processed/personal.train.txt"
systems_path = "processed/systems.train.txt"
tw_commercial_path = "processed/tw_commercial_group.train.txt"

articles_count = 237
corporate_count = 362
enron_t_s_count = 173
enron_travel_count = 19
hea_nesa_count = 79
personal_count = 159
systems_count = 109
tw_commercial_count = 1008
total_files = articles_count + corporate_count + enron_t_s_count + enron_travel_count + hea_nesa_count + personal_count + systems_count + tw_commercial_count


def main():
    nb = NaiveBayes.NaiveBayes()
    # 8 times add label
    nb.add_label("articles", "1.0", articles_count, articles_path)
    nb.add_label("corporate", "2.0", corporate_count, corporate_path)
    nb.add_label("enron_t_s", "3.0", enron_t_s_count, enron_t_s_path)
    nb.add_label("enron_travel_club", "4.0", enron_travel_count, enron_travel_path)
    nb.add_label("hea_nesa", "5.0", hea_nesa_count, hea_nesa_path)
    nb.add_label("personal", "6.0", personal_count, personal_path)
    nb.add_label("systems", "7.0", systems_count, systems_path)
    nb.add_label("tw_commercial_group", "8.0", tw_commercial_count, tw_commercial_path)

    nb.learn()
    print("Parameters learnt!")
    nb.decide()
    print("Decisions made! Check predictions.nb.txt in output folder")

if __name__ == '__main__':
    main()

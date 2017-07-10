__author__ = 'shail_000'

import Winnow as W
import Perceptron as P
import settings as S


perceptron_eta_list_margin = [1.5, 0.25, 0.03, 0.005, 0.001]
winnow_eta_list = [1.1, 1.01, 1.005, 1.0005, 1.0001]
winnow_margin_list = [2.0, 0.3, 0.04, 0.006, 0.001]
eta_p_1_c1 = 0.0
eta_p_1_c2 = 0.0
eta_w_0_1 = 0.0
eta_w_0_2 = 0.0
eta_w_1_1 = 0.0
gamma_w_1_1 = 0.0
eta_w_1_2 = 0.0
gamma_w_1_2 = 0.0


# takes a list of accuracies and finds the best accuracy, returns the index and best accuracy
def get_best_parameters(accuracy_list):
    max_accuracy = 0.0
    index = 0
    list_length = len(accuracy_list)
    for x in range(0, list_length):
        if accuracy_list[x] > max_accuracy:
            max_accuracy = accuracy_list[x]
            index = x
    return index, max_accuracy


# finds the best parameters for a given configuration for both Perceptron and Winnow
def tune_parameters():
    p_eta_list_length = len(perceptron_eta_list_margin)
    w_eta_list_length = len(winnow_eta_list)
    w_margin_list_length = len(winnow_margin_list)

    # Perceptron for learning rates C1
    accuracies1 = []
    global eta_p_1_c1
    for x in range(0, p_eta_list_length):
        weight_vector_n1_0 = S.create_initial_weight_list(S.n1)
        pr_margin_c1 = P.Perceptron(S.n1, S.m, weight_vector_n1_0, 0.0, perceptron_eta_list_margin[x], 1, S.config1_D1,
                                    S.config1_D2)
        pr_margin_c1.learn()
        pr_margin_c1.decide()
        accuracies1.append(pr_margin_c1.accuracy)

    index, acc_p_1_c1 = get_best_parameters(accuracies1)
    eta_p_1_c1 = perceptron_eta_list_margin[index]

    # Perceptron for learning rates C2
    accuracies2 = []
    global eta_p_1_c2
    for x in range(0, p_eta_list_length):
        weight_vector_n2_0 = S.create_initial_weight_list(S.n2)
        pr_margin_c2 = P.Perceptron(S.n2, S.m, weight_vector_n2_0, 0.0, perceptron_eta_list_margin[x], 1, S.config2_D1,
                                    S.config2_D2)
        pr_margin_c2.learn()
        pr_margin_c2.decide()
        accuracies2.append(pr_margin_c2.accuracy)

    index, acc_p_1_c2 = get_best_parameters(accuracies2)
    eta_p_1_c2 = perceptron_eta_list_margin[index]

    # Winnow gamma = 0 and eta list C1
    accuracies1 = []
    global eta_w_0_1
    for x in range(0, w_eta_list_length):
        weight_vector_n1_1 = S.create_initial_weight_list1(S.n1)
        wn_no_margin_c1 = W.Winnow(S.n1, S.m, weight_vector_n1_1, winnow_eta_list[x], 0, S.n1, S.config1_D1,
                                   S.config1_D2)
        wn_no_margin_c1.learn()
        wn_no_margin_c1.decide()
        accuracies1.append(wn_no_margin_c1.accuracy)

    index, acc_w_0_1 = get_best_parameters(accuracies1)
    eta_w_0_1 = winnow_eta_list[index]

    # Winnow gamma = 0 and eta list C2
    accuracies2 = []
    global eta_w_0_2
    for x in range(0, w_eta_list_length):
        weight_vector_n2_1 = S.create_initial_weight_list1(S.n2)
        wn_no_margin_c2 = W.Winnow(S.n2, S.m, weight_vector_n2_1, winnow_eta_list[x], 0, S.n2, S.config2_D1,
                                   S.config2_D2)
        wn_no_margin_c2.learn()
        wn_no_margin_c2.decide()
        accuracies2.append(wn_no_margin_c2.accuracy)

    index, acc_w_0_2 = get_best_parameters(accuracies2)
    eta_w_0_2 = winnow_eta_list[index]

    # Winnow gamma > 0 and eta list C1
    accuracies1 = []
    global eta_w_1_1, gamma_w_1_1
    for y in range(0, w_margin_list_length):
        for x in range(0, w_eta_list_length):
            weight_vector_n1_1 = S.create_initial_weight_list1(S.n1)
            wn_margin_c1 = W.Winnow(S.n1, S.m, weight_vector_n1_1, winnow_eta_list[x], winnow_margin_list[y],
                                    S.n1, S.config1_D1, S.config1_D2)
            wn_margin_c1.learn()
            wn_margin_c1.decide()
            accuracies1.append(wn_margin_c1.accuracy)

    index, acc_w_1_1 = get_best_parameters(accuracies1)
    best_margin_index = int(index/5)
    best_eta_index = int(index % 5)
    eta_w_1_1 = winnow_eta_list[best_eta_index]
    gamma_w_1_1 = winnow_margin_list[best_margin_index]

    # Winnow gamma > 0 and eta list C2
    accuracies2 = []
    global eta_w_1_2, gamma_w_1_2
    for y in range(0, w_margin_list_length):
        for x in range(0, w_eta_list_length):
            weight_vector_n2_1 = S.create_initial_weight_list1(S.n2)
            wn_margin_c2 = W.Winnow(S.n2, S.m, weight_vector_n2_1, winnow_eta_list[x], winnow_margin_list[y],
                                    S.n2, S.config2_D1, S.config2_D2)
            wn_margin_c2.learn()
            wn_margin_c2.decide()
            accuracies2.append(wn_margin_c2.accuracy)

    index, acc_w_1_2 = get_best_parameters(accuracies2)
    best_margin_index = int(index/5)
    best_eta_index = int(index % 5)
    eta_w_1_2 = winnow_eta_list[best_eta_index]
    gamma_w_1_2 = winnow_margin_list[best_margin_index]


if __name__ == '__main__':
    tune_parameters()
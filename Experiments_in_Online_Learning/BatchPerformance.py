__author__ = 'shail_000'

import Winnow as W
import Perceptron as P
import settings as S


perceptron_eta_list_margin = [1.5, 0.25, 0.03, 0.005, 0.001]
winnow_eta_list = [1.1, 1.01, 1.005, 1.0005, 1.0001]
winnow_margin_list = [2.0, 0.3, 0.04, 0.006, 0.001]


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


# finds the best parameters for the different configurations for both Perceptron and Winnow
def tune_parameters():
    p_eta_list_length = len(perceptron_eta_list_margin)
    w_eta_list_length = len(winnow_eta_list)
    w_margin_list_length = len(winnow_margin_list)

    # Perceptron
    weight_vector_n1_0 = S.create_initial_weight_list(1000)
    pr_no_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, 1.0, 0, S.D1m100, S.D2m100)
    pr_no_margin_c1.learn()
    pr_no_margin_c1.decide()
    acc = pr_no_margin_c1.accuracy
    print(acc)

    weight_vector_n1_0 = S.create_initial_weight_list(1000)
    pr_no_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, 1.0, 0, S.D1m500, S.D2m500)
    pr_no_margin_c1.learn()
    pr_no_margin_c1.decide()
    acc = pr_no_margin_c1.accuracy
    print(acc)

    weight_vector_n1_0 = S.create_initial_weight_list(1000)
    pr_no_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, 1.0, 0, S.D1m1000, S.D2m1000)
    pr_no_margin_c1.learn()
    pr_no_margin_c1.decide()
    acc = pr_no_margin_c1.accuracy
    print(acc)

    # Perceptron for learning rates change to 100, 500, 1000
    accuracies1 = []
    for x in range(0, p_eta_list_length):
        weight_vector_n1_0 = S.create_initial_weight_list(1000)
        pr_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, perceptron_eta_list_margin[x], 1.0, S.D1m100,
                                    S.D2m100)
        pr_margin_c1.learn()
        pr_margin_c1.decide()
        accuracies1.append(pr_margin_c1.accuracy)

    index, acc = get_best_parameters(accuracies1)
    eta_p_0_100 = perceptron_eta_list_margin[index]
    print(acc, eta_p_0_100)

    accuracies1 = []
    for x in range(0, p_eta_list_length):
        weight_vector_n1_0 = S.create_initial_weight_list(1000)
        pr_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, perceptron_eta_list_margin[x], 1.0, S.D1m500,
                                    S.D2m500)
        pr_margin_c1.learn()
        pr_margin_c1.decide()
        accuracies1.append(pr_margin_c1.accuracy)

    index, acc = get_best_parameters(accuracies1)
    eta_p_0_500 = perceptron_eta_list_margin[index]
    print(acc, eta_p_0_500)

    accuracies1 = []
    for x in range(0, p_eta_list_length):
        weight_vector_n1_0 = S.create_initial_weight_list(1000)
        pr_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, perceptron_eta_list_margin[x], 1.0, S.D1m1000,
                                    S.D2m1000)
        pr_margin_c1.learn()
        pr_margin_c1.decide()
        accuracies1.append(pr_margin_c1.accuracy)

    index, acc = get_best_parameters(accuracies1)
    eta_p_0_1000 = perceptron_eta_list_margin[index]
    print(acc, eta_p_0_1000)

    # Winnow gamma = 0 and eta list
    accuracies1 = []
    for x in range(0, w_eta_list_length):
        weight_vector_n1_1 = S.create_initial_weight_list1(1000)
        wn_no_margin_c1 = W.Winnow(1000, 5000, weight_vector_n1_1, winnow_eta_list[x], 0.0, 1000, S.D1m100, S.D2m100)
        wn_no_margin_c1.learn()
        wn_no_margin_c1.decide()
        accuracies1.append(wn_no_margin_c1.accuracy)

    index, acc = get_best_parameters(accuracies1)
    eta_w_0_100 = winnow_eta_list[index]
    print(acc, eta_w_0_100)

    accuracies1 = []
    for x in range(0, w_eta_list_length):
        weight_vector_n1_1 = S.create_initial_weight_list1(1000)
        wn_no_margin_c1 = W.Winnow(1000, 5000, weight_vector_n1_1, winnow_eta_list[x], 0.0, 1000, S.D1m500, S.D2m500)
        wn_no_margin_c1.learn()
        wn_no_margin_c1.decide()
        accuracies1.append(wn_no_margin_c1.accuracy)

    index, acc = get_best_parameters(accuracies1)
    eta_w_0_500 = winnow_eta_list[index]
    print(acc, eta_w_0_500)

    accuracies1 = []
    for x in range(0, w_eta_list_length):
        weight_vector_n1_1 = S.create_initial_weight_list1(1000)
        wn_no_margin_c1 = W.Winnow(1000, 5000, weight_vector_n1_1, winnow_eta_list[x], 0.0, 1000, S.D1m1000, S.D2m1000)
        wn_no_margin_c1.learn()
        wn_no_margin_c1.decide()
        accuracies1.append(wn_no_margin_c1.accuracy)

    index, acc = get_best_parameters(accuracies1)
    eta_w_0_1000 = winnow_eta_list[index]
    print(acc, eta_w_0_1000)

    # Winnow gamma > 0 and eta list
    accuracies1 = []
    for y in range(0, w_margin_list_length):
        for x in range(0, w_eta_list_length):
            weight_vector_n1_1 = S.create_initial_weight_list1(1000)
            wn_margin_c1 = W.Winnow(1000, 5000, weight_vector_n1_1, winnow_eta_list[x], winnow_margin_list[y],
                                    1000, S.D1m100, S.D2m100)
            wn_margin_c1.learn()
            wn_margin_c1.decide()
            accuracies1.append(wn_margin_c1.accuracy)

    index, acc = get_best_parameters(accuracies1)
    best_margin_index = int(index/5)
    best_eta_index = int(index % 5)
    eta_w_1_100 = winnow_eta_list[best_eta_index]
    gamma_w_1_100 = winnow_margin_list[best_margin_index]
    print(acc, eta_w_1_100, gamma_w_1_100)

    accuracies1 = []
    for y in range(0, w_margin_list_length):
        for x in range(0, w_eta_list_length):
            weight_vector_n1_1 = S.create_initial_weight_list1(1000)
            wn_margin_c1 = W.Winnow(1000, 50000, weight_vector_n1_1, winnow_eta_list[x], winnow_margin_list[y],
                                    1000, S.D1m500, S.D2m500)
            wn_margin_c1.learn()
            wn_margin_c1.decide()
            accuracies1.append(wn_margin_c1.accuracy)

    index, acc = get_best_parameters(accuracies1)
    best_margin_index = int(index/5)
    best_eta_index = int(index % 5)
    eta_w_1_500 = winnow_eta_list[best_eta_index]
    gamma_w_1_500 = winnow_margin_list[best_margin_index]
    print(acc, eta_w_1_500, gamma_w_1_500)

    accuracies1 = []
    for y in range(0, w_margin_list_length):
        for x in range(0, w_eta_list_length):
            weight_vector_n1_1 = S.create_initial_weight_list1(1000)
            wn_margin_c1 = W.Winnow(1000, 50000, weight_vector_n1_1, winnow_eta_list[x], winnow_margin_list[y],
                                    1000, S.D1m1000, S.D2m1000)
            wn_margin_c1.learn()
            wn_margin_c1.decide()
            accuracies1.append(wn_margin_c1.accuracy)

    index, acc = get_best_parameters(accuracies1)
    best_margin_index = int(index/5)
    best_eta_index = int(index % 5)
    eta_w_1_1000 = winnow_eta_list[best_eta_index]
    gamma_w_1_1000 = winnow_margin_list[best_margin_index]
    print(acc, eta_w_1_1000, gamma_w_1_1000)

    #train
    weight_vector_n1_0 = S.create_initial_weight_list(1000)
    pr_no_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, 1.0, 0, S.m100, S.m100_test)
    pr_no_margin_c1.learn()
    pr_no_margin_c1.decide()
    acc = pr_no_margin_c1.accuracy
    print(acc)
    weight_vector_n1_0 = S.create_initial_weight_list(1000)
    pr_no_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, 1.0, 0, S.m500, S.m500_test)
    pr_no_margin_c1.learn()
    pr_no_margin_c1.decide()
    acc = pr_no_margin_c1.accuracy
    print(acc)
    weight_vector_n1_0 = S.create_initial_weight_list(1000)
    pr_no_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, 1.0, 0, S.m1000, S.m1000_test)
    pr_no_margin_c1.learn()
    pr_no_margin_c1.decide()
    acc = pr_no_margin_c1.accuracy
    print(acc)
    weight_vector_n1_0 = S.create_initial_weight_list(1000)
    pr_no_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, eta_p_0_100, 0, S.m100, S.m100_test)
    pr_no_margin_c1.learn()
    pr_no_margin_c1.decide()
    acc = pr_no_margin_c1.accuracy
    print(acc)
    weight_vector_n1_0 = S.create_initial_weight_list(1000)
    pr_no_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, eta_p_0_500, 0, S.m500, S.m500_test)
    pr_no_margin_c1.learn()
    pr_no_margin_c1.decide()
    acc = pr_no_margin_c1.accuracy
    print(acc)
    weight_vector_n1_0 = S.create_initial_weight_list(1000)
    pr_no_margin_c1 = P.Perceptron(1000, 5000, weight_vector_n1_0, 0.0, eta_p_0_1000, 0, S.m1000, S.m1000_test)
    pr_no_margin_c1.learn()
    pr_no_margin_c1.decide()
    acc = pr_no_margin_c1.accuracy
    print(acc)
    #train Winnow gamma > 0 and eta list
    weight_vector_n1_1 = S.create_initial_weight_list1(1000)
    wn_margin_c1 = W.Winnow(1000, 50000, weight_vector_n1_1, eta_w_0_100, 0.0, 1000, S.m100, S.m100_test)
    wn_margin_c1.learn()
    wn_margin_c1.decide()
    print(wn_margin_c1.accuracy)
    weight_vector_n1_1 = S.create_initial_weight_list1(1000)
    wn_margin_c1 = W.Winnow(1000, 50000, weight_vector_n1_1, eta_w_0_500, 0.0, 1000, S.m500, S.m500_test)
    wn_margin_c1.learn()
    wn_margin_c1.decide()
    print(wn_margin_c1.accuracy)
    weight_vector_n1_1 = S.create_initial_weight_list1(1000)
    wn_margin_c1 = W.Winnow(1000, 50000, weight_vector_n1_1, eta_w_0_1000, 0.0, 1000, S.m1000, S.m1000_test)
    wn_margin_c1.learn()
    wn_margin_c1.decide()
    print(wn_margin_c1.accuracy)

    weight_vector_n1_1 = S.create_initial_weight_list1(1000)
    wn_margin_c1 = W.Winnow(1000, 50000, weight_vector_n1_1, eta_w_1_100, gamma_w_1_100, 1000, S.m100, S.m100_test)
    wn_margin_c1.learn()
    wn_margin_c1.decide()
    print(wn_margin_c1.accuracy)
    weight_vector_n1_1 = S.create_initial_weight_list1(1000)
    wn_margin_c1 = W.Winnow(1000, 50000, weight_vector_n1_1, eta_w_1_500, gamma_w_1_500, 1000, S.m500, S.m500_test)
    wn_margin_c1.learn()
    wn_margin_c1.decide()
    print(wn_margin_c1.accuracy)

    weight_vector_n1_1 = S.create_initial_weight_list1(1000)
    wn_margin_c1 = W.Winnow(1000, 50000, weight_vector_n1_1, eta_w_1_1000, gamma_w_1_1000, 1000, S.m1000, S.m1000_test)
    wn_margin_c1.learn()
    wn_margin_c1.decide()
    print(wn_margin_c1.accuracy)


if __name__ == '__main__':
    tune_parameters()
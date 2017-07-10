__author__ = 'shail_000'

import WinnowCurves as W
import PerceptronCurves as P
import settings as S
import matplotlib.pyplot as plt


perceptron_eta_list_margin = [1.5, 0.25, 0.03, 0.005, 0.001]
winnow_eta_list = [1.1, 1.01, 1.005, 1.0005, 1.0001]
winnow_margin_list = [2.0, 0.3, 0.04, 0.006, 0.001]


# takes lists, title and labels for curves and plots a graph with the name <name>
def plot_graphs(list1, list2, list3, list4, name, title, label1, label2, label3, label4):
    plt.figure()
    x_series = [40, 80, 120, 160, 200]
    plt.xlabel("Value of N")
    plt.ylabel("Mistakes")
    plt.title(title)
    plt.plot(x_series, list1, label=label1)
    plt.plot(x_series, list2, label=label2)
    plt.plot(x_series, list3, label=label3)
    plt.plot(x_series, list4, label=label4)
    plt.legend(loc="upper left")
    plt.savefig(name)


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


# finds the best parameters for the different configurations for both Perceptron and Winnow and plots the graph
def tune_parameters():
    p_eta_list_length = len(perceptron_eta_list_margin)
    w_eta_list_length = len(winnow_eta_list)
    w_margin_list_length = len(winnow_margin_list)

    # C1 Perceptron change values to 40, 80, 120, 160, 200
    weight_vector_n1_0 = S.create_initial_weight_list(200)
    pr_no_margin_c1 = P.Perceptron(200, 50000, weight_vector_n1_0, 0.0, 1.0, 0, S.n200, S.n200_test)
    pr_no_margin_c1.decide()
    acc = pr_no_margin_c1.accuracy
    mis = pr_no_margin_c1.mistakes

    # Perceptron for learning rates C1 change values 40, 80, 120, 160, 200
    weight_list_c1 = []
    bias_list_c1 = []
    mistakes_list_c1 = []
    accuracies1 = []
    for x in range(0, p_eta_list_length):
        weight_vector_n1_0 = S.create_initial_weight_list(200)
        pr_margin_c1 = P.Perceptron(200, 50000, weight_vector_n1_0, 0.0, perceptron_eta_list_margin[x], 1, S.n200,
                                    S.n200_test)
        pr_margin_c1.decide()
        weight_list_c1.append(pr_margin_c1.weight_vector)
        accuracies1.append(pr_margin_c1.accuracy)
        bias_list_c1.append(pr_margin_c1.bias)
        mistakes_list_c1.append(pr_margin_c1.mistakes)

    index, acc = get_best_parameters(accuracies1)
    bias = bias_list_c1[index]
    eta = perceptron_eta_list_margin[index]
    weight = weight_list_c1[index]
    mis = mistakes_list_c1[index]

    # Winnow gamma = 0 and eta list C1 change values to 40, 80, 120, 160, 200
    accuracies1 = []
    weight_list_c1 = []
    mistakes_list_c1 = []
    for x in range(0, w_eta_list_length):
        weight_vector_n1_1 = S.create_initial_weight_list1(200)
        wn_no_margin_c1 = W.Winnow(200, 50000, weight_vector_n1_1, winnow_eta_list[x], 0, 200, S.n200, S.n200_test)
        wn_no_margin_c1.decide()
        weight_list_c1.append(wn_no_margin_c1.weight_vector)
        accuracies1.append(wn_no_margin_c1.accuracy)
        mistakes_list_c1.append(wn_no_margin_c1.mistakes)

    index, acc = get_best_parameters(accuracies1)
    eta = winnow_eta_list[index]
    weight = weight_list_c1[index]
    mis = mistakes_list_c1[index]

    # Winnow gamma > 0 and eta list C2 <change values to 40, 80, 120, 160, 200
    accuracies2 = []
    weight_list_c2 = []
    mistakes_list_c2 = []
    for y in range(0, w_margin_list_length):
        for x in range(0, w_eta_list_length):
            weight_vector_n2_1 = S.create_initial_weight_list1(200)
            wn_margin_c2 = W.Winnow(200, 50000, weight_vector_n2_1, winnow_eta_list[x], 0, 200, S.n200, S.n200_test)
            wn_margin_c2.learn()
            wn_margin_c2.decide()
            weight_list_c2.append(wn_margin_c2.weight_vector)
            accuracies2.append(wn_margin_c2.accuracy)
            mistakes_list_c2.append(wn_margin_c2.mistakes)

    index, acc = get_best_parameters(accuracies2)
    best_margin_index = int(index/5)
    best_eta_index = int(index % 5)
    eta = winnow_eta_list[best_eta_index]
    weight = weight_list_c1[index]
    mis = mistakes_list_c1[index]
    gamma = winnow_margin_list[best_margin_index]

    # values obtained are manually filled into these lists, which can be verified from running the code
    list1 = [110, 87, 212, 40, 135]
    list2 = [77, 178, 150, 148, 61]
    list3 = [18, 45, 64, 75, 108]
    list4 = [34, 50, 64, 71, 47]
    plot_graphs(list1, list2, list3, list4, "FigC3.png", "Learning Curves", "P Margin 0", "P Margin 1", "W Margin 0",
                "W Margin !0")


if __name__ == '__main__':
    tune_parameters()
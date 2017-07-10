__author__ = 'shail_000'


import settings as S
import matplotlib.pyplot as plt
import Perceptron as P
import Winnow as W
import TuningParameters as T


# takes lists, title and labels for curves and plots a graph with the name <name>
def plot_graphs(list1, list2, list3, list4, name, title, label1, label2, label3, label4):
    plt.figure()
    x_series = S.create_iter_list()
    plt.xlabel("Number of instances")
    plt.ylabel("Cumulative Mistakes")
    plt.title(title)
    plt.plot(x_series, list1, label=label1)
    plt.plot(x_series, list2, label=label2)
    plt.plot(x_series, list3, label=label3)
    plt.plot(x_series, list4, label=label4)
    plt.legend(loc="upper left")
    plt.savefig(name)


# finds the number of mistakes for every configuration of Perceptron and Winnow and plots a graph of  cumulative
# mistakes and number of instances
def find_mistakes():
    T.tune_parameters()
    # C1 Perceptron
    weight_vector_n1_0 = S.create_initial_weight_list(S.n1)
    pr_no_margin_c1 = P.Perceptron(S.n1, S.m, weight_vector_n1_0, 0.0, 1.0, 0, S.config1_D1, S.config1_D2)
    pr_no_margin_c1.learn()
    pr_no_margin_c1.decide()
    print(pr_no_margin_c1.accuracy)
    mistakes_pr_c1_0 = pr_no_margin_c1.mistakes

    # Perceptron with hyper-parameters C1
    weight_vector_n1_0 = S.create_initial_weight_list(S.n1)
    pr_margin_c1 = P.Perceptron(S.n1, S.m, weight_vector_n1_0, 0.0, T.eta_p_1_c1, 1, S.config1_D1, S.config1_D2)
    pr_margin_c1.learn()
    pr_margin_c1.decide()
    print(pr_margin_c1.accuracy)
    mistakes_pr_c1_1 = pr_margin_c1.mistakes

    # Winnow with hyper-parameters C1 margin 0
    weight_vector_n1_1 = S.create_initial_weight_list1(S.n1)
    wn_no_margin_c1 = W.Winnow(S.n1, S.m, weight_vector_n1_1, T.eta_w_0_1, 0, S.n1, S.config1_D1, S.config1_D2)
    wn_no_margin_c1.learn()
    wn_no_margin_c1.decide()
    print(wn_no_margin_c1.accuracy)
    mistakes_wn_c1_0 = wn_no_margin_c1.mistakes

    # Winnow with hyper-parameters C1
    weight_vector_n1_1 = S.create_initial_weight_list1(S.n1)
    wn_margin_c1 = W.Winnow(S.n1, S.m, weight_vector_n1_1, T.eta_w_1_1, T.gamma_w_1_1, S.n1, S.config1_D1, S.config1_D2)
    wn_margin_c1.learn()
    wn_margin_c1.decide()
    print(wn_margin_c1.accuracy)
    mistakes_wn_c1_1 = wn_margin_c1.mistakes

    plot_graphs(mistakes_pr_c1_0, mistakes_pr_c1_1, mistakes_wn_c1_0, mistakes_wn_c1_1, "FigC1.png", "Config 1",
                "Perceptron C1 Margin 0", "Perceptron C1 Margin 1", "Winnow C1 Margin 0", "Winnow C1 Margin Non-zero")

    # Perceptron with hyper-parameters C2
    weight_vector_n2_0 = S.create_initial_weight_list(S.n2)
    pr_no_margin_c2 = P.Perceptron(S.n2, S.m, weight_vector_n2_0, 0.0, T.eta_p_1_c2, 1, S.config1_D2, S.config1_D2)
    pr_no_margin_c2.learn()
    pr_no_margin_c2.decide()
    print(pr_no_margin_c2.accuracy)
    mistakes_pr_c2_0 = pr_no_margin_c2.mistakes

    # Winnow with hyper-parameters C2 margin 0
    weight_vector_n2_1 = S.create_initial_weight_list1(S.n2)
    wn_no_margin_c2 = W.Winnow(S.n2, S.m, weight_vector_n2_1, T.eta_w_0_2, 0, S.n2, S.config2_D1, S.config2_D2)
    wn_no_margin_c2.learn()
    wn_no_margin_c2.decide()
    print(wn_no_margin_c2.accuracy)
    mistakes_pr_c2_1 = wn_no_margin_c2.mistakes

    # Winnow with hyper-parameters C2
    weight_vector_n2_1 = S.create_initial_weight_list1(S.n2)
    wn_margin_c2 = W.Winnow(S.n2, S.m, weight_vector_n2_1, T.eta_w_1_2, T.gamma_w_1_2, S.n2, S.config2_D1, S.config2_D2)
    wn_margin_c2.learn()
    wn_margin_c2.decide()
    print(wn_margin_c2.accuracy)
    mistakes_wn_c2_0 = wn_margin_c2.mistakes

    # C2 Perceptron
    weight_vector_n2_0 = S.create_initial_weight_list(S.n2)
    pr_no_margin_c2 = P.Perceptron(S.n2, S.m, weight_vector_n2_0, 0.0, 1.0, 0, S.config2_D2, S.config2_D2)
    pr_no_margin_c2.learn()
    pr_no_margin_c2.decide()
    print(pr_no_margin_c2.accuracy)
    mistakes_wn_c2_1 = pr_no_margin_c2.mistakes

    plot_graphs(mistakes_pr_c2_0, mistakes_pr_c2_1, mistakes_wn_c2_0, mistakes_wn_c2_1, "FigC2.png", "Config 2",
                "Perceptron C2 Margin 0", "Perceptron C2 Margin 1", "Winnow C2 Margin 0",
                "Perceptron C2 Margin Non-zero")

if __name__ == '__main__':
    find_mistakes()

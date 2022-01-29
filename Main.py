# -*- coding: utf-8 -*-
import numpy as np

from NeuralNetwork import NeuralNetwork
from Topology import Topology
from consts import *
from Kohonen import *
import matplotlib.pyplot as plt
import pandas
import sys

def get_data_set(rows_to_skip, rows_count):
    data_frame = pandas.read_csv('function.csv', delimiter=',', nrows=rows_count)
    array = numpy.array(data_frame['y'].dropna())
   # print(array)
    return array[::-1]


def normalize_and_divide(data, LEARN_PERCENTAGE):
   # data_list = {'min': min(data), 'max': max(data)}
  #  data_list['data'] = [(item - data_list['min']) / (data_list['max'] - data_list['min']) for item in data]
    last_learn_item = int(len(data) * LEARN_PERCENTAGE)

    return {
        'learn': data[:last_learn_item],
        'test': data[last_learn_item:]
    }


def learn_hybrid_kohonen_network(kohonen_layer_network, mlp_network, learn_data):
    print('Kohonen train:')
    kohonen_layer_network.learn(EPOCHS_KOHONEN)
    print('Normalized outputs from kohonen layer for train:')
    kohonen_layer_learn_outputs = kohonen_layer_network.get_normalize_outputs(learn_data)
    print(kohonen_layer_learn_outputs)

    print('MLP train')
    learn_error, rmse_arr = mlp_network.learn(kohonen_layer_learn_outputs)
    print('Train error: {}'.format(learn_error))
    plt.plot(rmse_arr)
    plt.show()
    return kohonen_layer_learn_outputs


def test_hybrid_kohonen_network(kohonen_layer_network, mlp_network, test_data):
    print('Normalized outputs from kohonen layer for test:')
    kohonen_layer_test_outputs = kohonen_layer_network.test(test_data)
    print(kohonen_layer_test_outputs)
    mse, prediction, real = mlp_network.test(kohonen_layer_test_outputs)
    print('Test error: %f' % mse)
    print(prediction)
    print(real)
    plt.plot(real, 'b')
    plt.plot(prediction, 'g')
    plt.show()
    return kohonen_layer_test_outputs

def train_and_test(learn_percentage=LEARN_PERCENTAGE, alpha_kohonen=ALPHA_KOHONEN,
                   neurons_count_kohonen=NEURONS_COUNT_KOHONEN, epochs=EPOCHS, window_size=WINDOW_SIZE):
    data = normalize_and_divide(get_data_set(ROWS_TO_SKIP, ROWS_COUNT), learn_percentage)
    kohonen_layer_network = Kohonen(data['learn'], alpha_kohonen, neurons_count_kohonen, window_size)
   # NN_LAYERS = [13, 8, neurons_count_kohonen]
    mlp_network = NeuralNetwork(Topology(NN_LAYERS))
    kohonen_layer_learn_outputs = learn_hybrid_kohonen_network(kohonen_layer_network, mlp_network, data['learn'], epochs)
    kohonen_layer_test_output, rmse = test_hybrid_kohonen_network(kohonen_layer_network, mlp_network, data['test'],
                                                                  window_size=window_size)
    return rmse

def test_learn_size():
    learn_size_min = 0.55
    learn_size_max = 0.95
    step = 0.05
    error_arr = []
    learns_size_arr = np.arange(learn_size_min, learn_size_max, step)
    #orig = consts.LEARN_PERCENTAGE
    for a in learns_size_arr:
       # consts.LEARN_PERCENTAGE = a
        print(a)
        test_error = train_and_test(learn_percentage=a)
        print(test_error)
        error_arr.append(test_error)
   # consts.LEARN_PERCENTAGE = orig
    return learns_size_arr, error_arr


if __name__ == '__main__':
    data = normalize_and_divide(get_data_set(ROWS_TO_SKIP, ROWS_COUNT), LEARN_PERCENTAGE)
    kohonen_layer_network = Kohonen(data['learn'], ALPHA_KOHONEN, NEURONS_COUNT_KOHONEN, WINDOW_SIZE)
    mlp_network = NeuralNetwork(Topology(NN_LAYERS))
    kohonen_layer_learn_outputs = learn_hybrid_kohonen_network(kohonen_layer_network, mlp_network, data['learn'])
    kohonen_layer_test_output = test_hybrid_kohonen_network(kohonen_layer_network, mlp_network, data['test'])
# if __name__ == "__main__":
#     data = get_data_from_framework()
#     data = normalize(data)
#
#     # Kohonen
#     print('Kohonen layer')
#     count_classes = 3
#     kohonen_layer_network = Kohonen(data['data'], ALPHA_KOHONEN, count_classes, len(data['data'][0]))
#     error = kohonen_layer_network.train(EPOCHS_KOHONEN)
#     target = kohonen_layer_network.test(data['data'])
#     target_classes = get_target_classes(target)
#     plt.plot(error)
#     plt.show()
#
#     # MLP
#     print('MLP')
#     data['target'] = target_classes
#     data = divide_data(data, LEARN_PERCENTAGE)
#     topology = Topology(NN_LAYERS)
#     nn = NeuralNetwork(topology)
#     learn_error, loss_arr = nn.learn(data)
#     test_error = nn.test(data)
#     print(learn_error)
#     print(test_error)
#     plt.plot(loss_arr)
#     plt.show()    # learn_size, error_arr = test_learn_size()
#     # plt.xlabel('learn_size')
#     # plt.ylabel('error')
#     # plt.plot(learn_size, error_arr)
#     # plt.show()


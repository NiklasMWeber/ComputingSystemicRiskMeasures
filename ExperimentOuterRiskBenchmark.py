from datetime import datetime
import itertools
import os
import copy
import math
import time
from multiprocessing import Pool

import dgl
import numpy as np
import torch
import torch.nn as nn

import Models
from EisenbergNoe import get_clearing_vector_iter_from_batch
from DataGeneration import DataGenerator, get_graphs_from_list

# This file deals with finding the systemic risk of a financial network by searching for the smallest amount of capital
# that secures the system, when the capital is allocated optimally.
# Running the main function starts the capital search of all parameter combinations from the "config" dictionary.
# This is the file for obtaining the results of the Benchmarks that do not need training!

# get package-, output-path and timestamp
path_to_package = os.path.abspath(os.getcwd()).split('ComputingSystemicRiskMeasures')[0] + 'ComputingSystemicRiskMeasures/'
dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_path = path_to_package + 'Experiments/' + dt_string + '/'


# config dictionary. Specify which dataset, model, and parameter combinations you want to train
config = {
    # Hand over path and time stamp for saving trained models, plots, etc...
    'path_to_package': [path_to_package],
    'out_path': [out_path],
    'dt_string': [dt_string],
    # Choose datasets to be trained on
    'dataset': ['ER', 'CP', 'CPf'],

    'nn_specs':  [('Uniform',), ('Default',), ('Level1',)],

    'number_of_epochs': [100],
    'batch_size': [10],  # [1, 10, 100],
    'learning_rate': [1e-3],
    'seed': [1900, 1954, 1974, 1990, 2014],  # although benchmarks are deterministic, probabilistic bisection search is random --> use seeds
}

# you can add parameter combination manually by providing
# path_to_package, out_path, dt_string, data, nn_specs, epochs, batch_size, lr, seed
manual_exp_configs = None  # e.g. [(path_to_package, out_path, dt_string, 'ER', ('GNN', 10, 10, 1, 1, 1), 1, 33, 0.002, 1900)]


# this method writes arrays to a log file
def write_arrays_to_file(arrays, filename):
    combined_arrays = []
    for array in arrays:
        combined_arrays += array + [['']]
    with open(filename, 'w') as file:
        for array in combined_arrays:
            file.write('\t'.join(array) + '\n')

# this printer class prints and stores messages during training
class ExperimentPrinter:

    def __init__(self, start_time, exp_id, out_path):
        self.start_time = start_time
        self.exp_id = exp_id
        self.out_path = out_path
        self.arrays = []

    def print(self, array_to_add):
        # prints, write to print file and returns appended array
        array = ['{:06.0f}'.format(time.time() - self.start_time), '{:3.0f}'.format(self.exp_id), *array_to_add]
        print('\t'.join(array))
        file = open(self.out_path + 'exp_print_log.txt', 'a')
        file.write('\t'.join(array) + '\n')
        file.close()
        self.arrays.append(array)


# helper function to determine next capital level in probabilistic bisection search
def next_root_idx(func_val, current_idx, current_dist, values):
    p = 0.75
    q = 1-p
    if func_val < 0:  # val < 0 --> root is smaller --> we want more weight on smaller values
        current_dist[current_idx:] = 2*q*current_dist[current_idx:]  # mult with 2*q decreases prob of higher val
        current_dist[:current_idx] = 2*p*current_dist[:current_idx]  # mult with 2*p increases prob of lower val
    else:  # func_val > 0 --> root is bigger --> want more weight on higher values
        current_dist[current_idx:] = 2 * p * current_dist[current_idx:]  # mult with 2*p increases prob of higher val
        current_dist[:current_idx] = 2 * q * current_dist[:current_idx]  # mult with 2*q decreases prob of lower val

    current_dist = current_dist/current_dist.sum()
    next_idx = 0
    cdf = current_dist[0]
    while cdf < 0.5:
        next_idx += 1
        cdf += current_dist[next_idx]
    next_root = values[next_idx]
    return next_idx, next_root, current_dist


def risk_func(x):
    return torch.log(torch.exp(0.01*x).mean())/0.01
    # return torch.mean(x)  # expectation risk measure


def evaluate(graph,
             list_of_graphs,
             batch_of_liab,
             batch_of_assets,
             batch_of_outgoing,
             model: nn.Module,
             bailout_capital,
             ) -> float:
    bailout_ratio = model.predict(graph, list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing)
    batch_of_cv = get_clearing_vector_iter_from_batch(batch_of_liab,
                                                      batch_of_assets + bailout_ratio.unsqueeze(-1) * bailout_capital,
                                                      batch_of_outgoing, n_iter=50)
    batch_of_node_loss = batch_of_outgoing - batch_of_cv
    batch_of_graph_loss = batch_of_node_loss.sum(1)  # first dim batch, second dim node
    mean_loss = risk_func(batch_of_graph_loss)
    return mean_loss


def train(train_data, # train data input not used by default since train data is sampled randomly
          val_data,
          test_data,
          model,
          data_generator,
          max_bailout_capital: float,
          max_accept_risk: float,
          number_of_epochs: int,
          batch_size: int,
          learning_rate: float = 0.01,
          exp_pr: ExperimentPrinter = None):

    # initialize probabilistic bisection search for optimal bailout capital
    start = 0
    stop = max_bailout_capital
    step = 0.0001
    values = np.arange(start=start, stop=stop, step=step)
    dist = np.ones(values.shape) / len(values)
    current_idx = int(len(values) / 2)
    current_root = values[current_idx]

    # set number_of_grad_steps. Determines how grad steps are performed in each epoch. 1000 points are sampled per epoch for training
    number_of_grad_steps = 1000 // batch_size  # always want to train on same amount of data. bigger batches -> fewer steps

    # Init bailout and optimizer
    bailout_capital = torch.Tensor([current_root])
    best_model = model
    last_model = model

    train_data = data_generator.get_graphs(100)  # initialize train data for 0 bailout information
    # val_data = data_generator.get_graphs(2500)  # can be used to initialize val/test randomly
    # test_data = data_generator.get_graphs(2500) # can be used to initialize val/test randomly

    train_performance = evaluate(dgl.batch(train_data[0]), train_data[0], train_data[1][0], train_data[1][1], train_data[1][2],  best_model, torch.Tensor([0.0]))
    val_performance = evaluate(dgl.batch(val_data[0]), val_data[0], val_data[1][0], val_data[1][1], val_data[1][2], best_model, torch.Tensor([0.0]))
    test_performance = evaluate(dgl.batch(test_data[0]), test_data[0], test_data[1][0], test_data[1][1], test_data[1][2], best_model, torch.Tensor([0.0]))

    exp_pr.print(['Epoch', '{:05d}'.format(0), 'Train Risk','{:6.2f}'.format(train_performance),
                  'Val Risk', '{:6.2f}'.format(val_performance), 'Test Risk', '{:6.2f}'.format(test_performance),
                  'Capital', '{:4.2f}'.format(0)])

    for epoch in range(1, number_of_epochs + 1):

        # Set bailout capital to current_root of probabilistic bisection search
        # In first epoch this has no effect. In following epochs it will set the capital to the level determined at end of previous epoch...
        bailout_capital = torch.Tensor([current_root])
        list_of_batch_graph_losses = []

        ################################################################################################################
        # for training on random data
        list_of_all_graphs = []
        for i in range(number_of_grad_steps):
            list_of_graphs, useful_tensor = data_generator.get_graphs(batch_size)
            list_of_all_graphs += list_of_graphs
        ################################################################################################################

        ################################################################################################################
        # for training on given data
        # list_of_batches = get_list_of_batches_from_list_of_graphs(number_of_grad_steps, train_data[0])
        # for list_of_graphs in list_of_batches:
        #     list_of_graphs, useful_tensor = get_graphs_from_list(list_of_graphs)
        ################################################################################################################
            (batch_of_liab, batch_of_assets, batch_of_outgoing_liab) = useful_tensor

            # calculate loss
            bailout_ratio = model.predict(dgl.batch(list_of_graphs), list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing_liab)
            batch_of_cv = get_clearing_vector_iter_from_batch(batch_of_liab,
                                                              batch_of_assets + bailout_ratio.unsqueeze(-1) * bailout_capital,
                                                              batch_of_outgoing_liab, n_iter=50)
            batch_of_node_loss = batch_of_outgoing_liab - batch_of_cv
            batch_of_graph_loss = batch_of_node_loss.sum(1)  # first dim batch, second dim node

            list_of_batch_graph_losses.append(batch_of_graph_loss)


        rolling_train_performance = risk_func(torch.stack(list_of_batch_graph_losses))
        train_performance = rolling_train_performance

        val_performance = evaluate(dgl.batch(val_data[0]), val_data[0], val_data[1][0], val_data[1][1],
                                           val_data[1][2],  model, bailout_capital)

        last_model = copy.deepcopy(model)


        if epoch % 1 == 0:
            test_performance = evaluate(dgl.batch(test_data[0]), test_data[0], test_data[1][0], test_data[1][1],
                                            test_data[1][2],  model, bailout_capital)

            exp_pr.print(['Epoch', '{:05d}'.format(epoch), 'Train Risk', '{:6.2f}'.format(train_performance),
                          'Val Risk', '{:6.2f}'.format(val_performance), 'Test Risk', '{:6.2f}'.format(test_performance),
                          'Capital', '{:4.2f}'.format(bailout_capital.item())])

        # calculate next bailout capital from probabilistic bisection search
        func_val = train_performance - max_accept_risk
        current_idx, current_root, dist = next_root_idx(func_val, current_idx, dist, values)
        # bailout capital is updated is not updated here, but at the start of next epoch


    train_performance = evaluate(dgl.batch(train_data[0]), train_data[0], train_data[1][0], train_data[1][1],
                                 train_data[1][2], model, bailout_capital)
    val_performance = evaluate(dgl.batch(val_data[0]), val_data[0], val_data[1][0], val_data[1][1], val_data[1][2],
                              model, bailout_capital)
    test_performance = evaluate(dgl.batch(test_data[0]), test_data[0], test_data[1][0], test_data[1][1],
                                test_data[1][2], model, bailout_capital)

    exp_pr.print(['Final', '{:05d}'.format(100), 'Train Risk', '{:6.2f}'.format(train_performance),
                  'Val Risk', '{:6.2f}'.format(val_performance), 'Test Risk', '{:6.2f}'.format(test_performance),
                  'Capital', '{:4.2f}'.format(bailout_capital.item())])

    return bailout_capital.item(), exp_pr, last_model

# search method does some overhead tasks (load data, initialize model,...) and starts training for each parameter combination
def search(var: list):
    (path_to_package, out_path, dt_string, dataset, nn_specs, number_of_epochs,batch_size,
     learning_rate, seed, exp_id) = var

    start_time = int(time.time())
    exp_pr = ExperimentPrinter(start_time=start_time, exp_id=exp_id, out_path=out_path)

    exp_pr.print(['Info', 'Timestamp', '{:10.0f}'.format(start_time)])
    exp_pr.print(['Info', 'Datestring', dt_string])
    exp_pr.print(['Info', 'Output path', out_path])
    exp_pr.print(['Info', 'Datatype', dataset])

    if dataset == 'CP':
        filename = 'Data/5000_beta_assetFactor_50_10_shuffled.bin'  # 'Data/50000_beta_society_assetFactor_50_10_noSoc.bin'
    elif dataset == 'CPf':
        filename = 'Data/5000_beta_assetFactor_50_10_fixed_idx0_unshuffled.bin'  # 'Data/50000_beta_society_assetFactor_50_10_fixed_idx0_noSoc.bin'
    elif dataset == 'ER':
        filename = 'Data/5000_beta_assetFactor_erdos_renyi_10_04.bin'  # 'Data/50000_beta_society_assetFactor_erdos_renyi_10_04_noSoc.bin'
    else:
        raise AttributeError
    data_path = path_to_package + filename
    data_generator = DataGenerator(dataset, path_to_package + 'Data/5000_beta_assetFactor_50_10_fixed_idx0_unshuffled[0].bin')

    train_ratio = 0.0
    val_ratio = 0.5
    test_ratio = 0.5
    max_bailout_capital = 500
    max_accept_risk = 100
    if seed is None:
        seed = np.random.randint(0, 3000)
    np.random.seed(seed)
    torch.manual_seed(seed)

    exp_pr.print(['Info', 'Data path', data_path])
    exp_pr.print(['Info', 'Model', str(nn_specs)])
    exp_pr.print(['Info', 'Maximal Bailout', str(max_bailout_capital)])
    exp_pr.print(['Info', 'Val/Test Ratios', str(val_ratio), str(test_ratio)])
    exp_pr.print(['Info', 'Epochs', str(number_of_epochs)])
    exp_pr.print(['Info', 'Batch Size', str(batch_size)])
    exp_pr.print(['Info', 'Learning Rate', str(learning_rate)])
    exp_pr.print(['Info', 'Seed', str(seed)])


    exp_pr.print(['Step', 'Loading data'])
    list_of_graphs, dict_of_labels = dgl.load_graphs(data_path)

    exp_pr.print(['Step', 'Preprocessing'])

    train_idx = int(len(list_of_graphs)*train_ratio)
    val_idx = int(len(list_of_graphs)*val_ratio)
    # train_data = DataGeneration.get_graphs_from_list(list_of_graphs[:train_idx]) # can load fixed train data here if not training on randomly sampled data
    val_data = get_graphs_from_list(list_of_graphs[train_idx:train_idx+val_idx])
    test_data = get_graphs_from_list(list_of_graphs[train_idx+val_idx:])

    # Create model and train
    exp_pr.print(['Step', 'Initialize Model'])

    if nn_specs[0] == 'Uniform':
        model = Models.UniformBailoutPredictor()
    elif nn_specs[0] == 'Default':
        model = Models.DefaultOnlyBailoutPredictor()
    else:
        model = Models.FirstLevelBailoutPredictor()

    train_start = time.time()
    exp_pr.print(['Step', 'Training'])

    final_bailout_capital, exp_pr, model = train(train_data=None,
                                                                       val_data=val_data,
                                                                       test_data=test_data,
                                                                       model=model,
                                                                       data_generator = data_generator,
                                                                       max_bailout_capital = max_bailout_capital,
                                                                       max_accept_risk = max_accept_risk,
                                                                       number_of_epochs=number_of_epochs,
                                                                       batch_size=batch_size,
                                                                       learning_rate=learning_rate,
                                                                       exp_pr=exp_pr)


    exp_pr.print(['Step', 'Save Model'])

    duration = time.time() - train_start
    exp_pr.print(['Info','Duration', '{:.4f}'.format(duration)])
    print()
    return exp_pr.arrays


if __name__ == '__main__':

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # combinatorics
    keys = list(config)
    param_comb = [comb for comb in itertools.product(*map(config.get, keys))]

    if manual_exp_configs is not None:
        param_comb += manual_exp_configs

    ####################################################################################################################
    # multiprocessing
    # param_comb_multi = [[*comb,id] for id, comb in enumerate(param_comb)]
    # pool = Pool(16)
    # exp_arrays = pool.map(search, param_comb_multi)
    ####################################################################################################################

    ####################################################################################################################
    # debugging loop (can be used instead of multiprocessing)
    exp_arrays = []
    for exp_id, comb in enumerate(param_comb):
        exp_array = search([*comb, exp_id])
        exp_arrays.append(exp_array)
    ####################################################################################################################

    write_arrays_to_file(exp_arrays, out_path + 'exp_log.txt')
    print('Done')

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

# This file deals with decreasing the systemic risk of a financial network by learning to allocate a fixed amount of
# bailout capital optimally.
# Running the main function starts the train runs of all parameter combinations from the "config" dictionary.
# This is the file for obtaining th results of the Benchmarks that do not need training!


# get package-, output-path and timestamp
path_to_package = os.path.abspath(os.getcwd()).split('ComputingSystemicRiskMeasures')[0] + 'ComputingSystemicRiskMeasures/'
dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_path = path_to_package + 'Experiments/' + dt_string + '/'

# config dictionary. Specify which dataset, model, and parameter combinations you want to train
config = {
    # Hand over path and time stamp for saving trained models and log files
    'path_to_package': [path_to_package],
    'out_path': [out_path],
    'dt_string': [dt_string],
    # Choose datasets to be trained on
    'dataset': ['ER','CP', 'CPf'],

    'nn_specs':  [('Uniform',), ('Default',), ('Level1',)],

    'number_of_epochs': [1],  # doe snot need multiple epochs since no training is required for benchmarks
    'batch_size': [10],  # [1, 10, 100],
    'learning_rate': [1e-3],
    'seed': [1900], # 1900, 1954, 1974, 1990, 2014
    # [1e-3, 1e-4, 1e-5],  # 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001,    , 0.0001, 0.00001, 0.000001, 0.0000001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8
}

# path_to_package, out_path, dt_string, data, nn_specs, epochs, batch_size, lr, seed
manual_exp_configs = None  # [(path_to_package, out_path, dt_string, 'ER', ('GNN', 10, 10, 1, 1, 1), 1, 33, 0.002, 1900)]


def write_arrays_to_file(arrays, filename):
    combined_arrays = []
    for array in arrays:
        combined_arrays += array + [['']]
    with open(filename, 'w') as file:
        for array in combined_arrays:
            file.write('\t'.join(array) + '\n')

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



# def AVaR(level: float, data: torch.tensor):
#     data = torch.sort(data, dim=0, descending=True).values
#     border_idx = int(len(data) * level)
#     return torch.mean(data[:-border_idx])

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


def train(train_data,
          val_data,
          test_data,
          model,
          data_generator,
          bailout_capital: float,
          number_of_epochs: int,
          batch_size: int,
          learning_rate: float = 0.01,
          exp_pr: ExperimentPrinter = None):

    # Init bailout and optimizer
    bailout_capital = torch.Tensor([bailout_capital])
    best_model = model
    last_model = model

    train_performance = 0  # evaluate(dgl.batch(train_data[0]), train_data[0], train_data[1][0], train_data[1][1], train_data[1][2],  best_model, torch.Tensor([0.0]))
    val_performance = evaluate(dgl.batch(val_data[0]), val_data[0], val_data[1][0], val_data[1][1], val_data[1][2], best_model, torch.Tensor([0.0]))
    test_performance = evaluate(dgl.batch(test_data[0]), test_data[0], test_data[1][0], test_data[1][1], test_data[1][2], best_model, torch.Tensor([0.0]))

    exp_pr.print(['Epoch', '{:05d}'.format(0), 'Train Risk','{:6.2f}'.format(train_performance),
                  'Val Risk', '{:6.2f}'.format(val_performance), 'Test Risk', '{:6.2f}'.format(test_performance),
                  'Capital', '{:4.2f}'.format(0)])

    for epoch in range(1, number_of_epochs + 1):
        train_performance = 0

        val_performance = evaluate(dgl.batch(val_data[0]), val_data[0], val_data[1][0], val_data[1][1],
                                           val_data[1][2],  model, bailout_capital)

        test_performance = evaluate(dgl.batch(test_data[0]), test_data[0], test_data[1][0], test_data[1][1],
                                            test_data[1][2],  model, bailout_capital)

        exp_pr.print(['Final', '{:05d}'.format(epoch), 'Train Risk', '{:6.2f}'.format(train_performance),
                          'Val Risk', '{:6.2f}'.format(val_performance), 'Test Risk', '{:6.2f}'.format(test_performance),
                          'Capital', '{:4.2f}'.format(bailout_capital.item())])


    return best_model, bailout_capital.item(), exp_pr, last_model

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
    initial_bailout_capital = 50.0
    if seed is None:
        seed = np.random.randint(0, 3000)
    np.random.seed(seed)
    torch.manual_seed(seed)

    exp_pr.print(['Info', 'Data path', data_path])
    exp_pr.print(['Info', 'Model', str(nn_specs)])
    exp_pr.print(['Info', 'Initial Bailout', str(initial_bailout_capital)])
    exp_pr.print(['Info', 'Val/Test Ratios', str(val_ratio), str(test_ratio)])
    exp_pr.print(['Info', 'Epochs', str(number_of_epochs)])
    exp_pr.print(['Info', 'Batch Size', str(batch_size)])
    exp_pr.print(['Info', 'Learning Rate', str(learning_rate)])
    exp_pr.print(['Info', 'Seed', str(seed)])

    exp_pr.print(['Step', 'Loading data'])
    list_of_graphs, dict_of_labels = dgl.load_graphs(data_path)
    # list_of_graphs = list_of_graphs[:500]

    exp_pr.print(['Step', 'Preprocessing'])

    train_idx = int(len(list_of_graphs)*train_ratio)
    val_idx = int(len(list_of_graphs)*val_ratio)
    # train_data = DataGeneration.get_graphs_from_list(list_of_graphs[:train_idx])
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

    model, final_bailout_capital, exp_pr, last_model = train(train_data=None,
                                                                       val_data=val_data,
                                                                       test_data=test_data,
                                                                       model=model,
                                                                       data_generator = data_generator,
                                                                       bailout_capital=initial_bailout_capital,
                                                                       number_of_epochs=number_of_epochs,
                                                                       batch_size=batch_size,
                                                                       learning_rate=learning_rate,
                                                                       exp_pr=exp_pr)


    exp_pr.print(['Step', 'Save Model'])

    # model_name = ''
    # for comp in nn_specs:
    #     model_name = model_name + f'{comp}_'
    # model_name = model_name + f'{dataset}_{learning_rate}_{batch_size}_{exp_id}'
    # torch.save(model.state_dict(), out_path + model_name + '.pt')
    # torch.save(last_model.state_dict(), out_path + model_name + '_last.pt')

    duration = time.time() - train_start
    exp_pr.print(['Info','Duration', '{:.4f}'.format(duration)])
    print()
    return exp_pr.arrays


########################################################################################################################
########################################################################################################################
# Original main from experiment


if __name__ == '__main__':

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # combinatorics
    keys = list(config)
    # param_comb = itertools.product(*map(config.get, keys))
    param_comb = [comb for comb in itertools.product(*map(config.get, keys))]

    if manual_exp_configs is not None:
        param_comb += manual_exp_configs

    # multiprocessing  ### currently not running on laptop environment... ###
    param_comb_multi = [[*comb,id] for id, comb in enumerate(param_comb)]
    pool = Pool(6)
    exp_arrays = pool.map(search, param_comb_multi)

    # debugging loop (can be used isntead of multiprocessing)
    # exp_arrays = []
    # for exp_id, comb in enumerate(param_comb):
    #     exp_array = search([*comb, exp_id])
    #     exp_arrays.append(exp_array)

    write_arrays_to_file(exp_arrays, out_path + 'exp_log.txt')
    print('Done')

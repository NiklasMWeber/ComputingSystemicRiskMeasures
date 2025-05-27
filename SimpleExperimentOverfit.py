import time
import os
from datetime import datetime, timedelta

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv


# 2* N graphs with i=0,...,N-1 for cascade and star network

# get package-, output-path and timestamp
path_to_package = os.path.abspath(os.getcwd()).split('GNN')[0] + 'GNN/LearnRiskMeasure/'
dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_path = path_to_package + 'Experiments/' + dt_string + '_Overfit/'

def my_print(message):
    print(message)
    file = open(out_path + 'exp_log.txt', 'a')
    file.write(message + '\n')
    file.close()

def get_liability_matrix(g, weight_name='weight'):
    # get edge source and destination
    src, dst = g.edges()
    # initialize zeros matrix
    liab = torch.zeros(g.num_src_nodes(), g.num_dst_nodes())
    # fill entries according to edges with corresponding edata
    liab[src, dst] = g.edata[weight_name].squeeze(-1)
    return liab

def get_clearing_vector_iter(graph, bailout, n_iter = 100):  # batch_of_liab:Tensor, batch_of_assets:Tensor, batch_of_outgoing_liab:Tensor, n_iter: int = 50):
    assets = graph.ndata['assets'] + bailout
    liab = get_liability_matrix(graph)
    outgoing = liab.sum(dim=-1)
    cv = assets * 0
    pi_trp = torch.nan_to_num(liab / outgoing, nan=0, posinf=0, neginf=0).transpose(-2,-1)
    for i in range(n_iter):
        cv_new = assets + pi_trp @ cv
        cv_new = torch.minimum(cv_new, outgoing)
        if abs(cv-cv_new).sum() < 1e-7:
            return cv_new
        cv = cv_new
    return cv

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.layer1 = SAGEConv(1, 10, 'mean')
        self.layer2 = SAGEConv(10, 1, 'mean')

    def forward(self, g, in_feat, e_feat):
        h = self.layer1(g, in_feat, e_feat)
        h = torch.sigmoid(h)
        h = self.layer2(g, h, e_feat)
        return h

    def predict(self, graph):
        features = graph.ndata['assets'].reshape(-1, 1)
        efeat = torch.reshape(graph.edata['weight'], (-1, 1))
        bailout_score = self.forward(graph, features, efeat).reshape(-1)
        bailout_score = torch.softmax(bailout_score, dim=0)
        return bailout_score


class XPENN(nn.Module):
    def __init__(self):
        super(XPENN, self).__init__()
        self.base_layer_dim = 10
        self.node_feat_size = 1

        self.phi1 = nn.Linear(1 + 2 * self.node_feat_size, self.base_layer_dim)
        self.alpha1 = nn.Linear(self.node_feat_size + self.base_layer_dim, self.base_layer_dim)
        self.rho1 = nn.Linear(self.node_feat_size + self.base_layer_dim*2, self.base_layer_dim)
        self.rho2 = nn.Linear(self.base_layer_dim, 1)
        self.psi1 = nn.Linear(2 + 2 * self.node_feat_size, self.base_layer_dim)
        self.psi2 = nn.Linear(self.base_layer_dim, self.base_layer_dim)

    def forward(self, original_node_feat, liab_matrix_tensor):
        original_node_feat_size = original_node_feat.size()
        number_of_nodes = original_node_feat_size[-2]
        repeat_dim = [1 for dim in original_node_feat_size]
        repeat_dim.append(number_of_nodes)
        reshape_dim = list(original_node_feat_size)
        reshape_dim.insert(-2, number_of_nodes)
        node_feat_src = original_node_feat.repeat(repeat_dim).reshape(reshape_dim)
        node_feat_dst = node_feat_src.transpose(dim0=-3, dim1=-2)
        stacked_features = torch.cat([node_feat_src, liab_matrix_tensor.unsqueeze(dim=-1), node_feat_dst], dim=-1)

        stacked_features_for_psi = torch.cat([node_feat_src, liab_matrix_tensor.unsqueeze(dim=-1),
                                              torch.transpose(liab_matrix_tensor, dim0=0, dim1=1).unsqueeze(dim=-1),
                                              node_feat_dst], dim=-1)

        h_psi = self.psi1(stacked_features_for_psi)
        h_psi = torch.sigmoid(h_psi)
        h_psi = self.psi2(h_psi)
        h_psi = h_psi.mean(dim=-2)

        h = self.phi1(stacked_features)
        h = h.mean(dim=-2)  # sum messages (dim=-1) over all dst j (-2) for a fixed src i (-3)

        h = torch.cat([original_node_feat, h], dim=-1)
        h = self.alpha1(h)

        h = h.mean(dim=-2)  # sum messages (dim=-1) of all nodes (dim=-2)

        h = h.unsqueeze(dim=-2)
        repeat_dim = [1 for _ in h.size()]  # torch.ones(len(original_node_feat_size))
        repeat_dim[-2] = number_of_nodes
        h = h.repeat(repeat_dim)
        h = torch.cat([original_node_feat, h, h_psi], dim=-1)

        h = self.rho1(h)
        h = torch.sigmoid(h)
        h = self.rho2(h)

        return h

    def predict(self, graph):
        features_per_node = graph.ndata['assets']
        # tmp_id = np.arange(N)
        # numeration = torch.tensor(list(tmp_id))
        features_per_node = torch.stack([features_per_node], dim=1)

        liab = get_liability_matrix(graph)

        bailout_score = self.forward(features_per_node, liab).reshape(-1)
        bailout_score = torch.softmax(bailout_score, dim=0)
        return bailout_score


class NNL(nn.Module):
    def __init__(self, N):
        super(NNL, self).__init__()
        self.layer1 = nn.Linear(N + N * N, N)  # int(np.sqrt(N)))
        self.layer2 = nn.Linear(N, N)  # int(np.sqrt(N)), N)

    def forward(self, in_feat):
        h = self.layer1(in_feat)
        h = torch.sigmoid(h)
        h = self.layer2(h)
        return h

    def predict(self, graph):
        features = graph.ndata['assets'].reshape(1, -1)
        liab = get_liability_matrix(graph).reshape(1, -1)
        features_with_liab = torch.concat((features, liab), dim=1)
        bailout_score = self.forward(features_with_liab).reshape(-1)
        bailout_score = torch.softmax(bailout_score, dim=0)
        return bailout_score

def generate_data(N):
    list_of_graphs = []
    for base_node_id in range(N):
        # graph 1
        src = [int(i) for i in np.arange(N)]
        dst = [int((i + 1) % N) for i in src]

        del_idx = (base_node_id - 1) % N
        del src[del_idx], dst[del_idx]

        edges = torch.LongTensor(src), torch.LongTensor(dst)
        weights = torch.Tensor(np.ones((N - 1,))) * (N - 1.0)  # weight of each edge
        g_1 = dgl.graph(edges)
        g_1.edata['weight'] = weights + 1
        g_1.ndata['assets'] = torch.Tensor(np.zeros(N)) + 1

        liab = get_liability_matrix(g_1)
        g_1.ndata['outgoingLiabilities'] = liab.sum(-1)
        g_1.ndata['incomingLiabilities'] = liab.sum(-2)

        list_of_graphs.append(g_1)

        # graph 2

        src = [int(i) for i in np.arange(N)]
        dst = [int(base_node_id) for i in np.arange(N)]

        del_idx = base_node_id
        del src[del_idx], dst[base_node_id]

        edges = torch.LongTensor(src), torch.LongTensor(dst)
        weights = torch.Tensor(np.ones((N - 1,))) + 1  # weight of each edge
        g_2 = dgl.graph(edges)
        g_2.edata['weight'] = weights
        g_2.ndata['assets'] = torch.Tensor(np.zeros(N)) + 1

        # en = EisenbergNoe.EisenbergNoe(1.0, 1.0)
        liab = get_liability_matrix(g_2)
        g_2.ndata['outgoingLiabilities'] = liab.sum(-1)
        g_2.ndata['incomingLiabilities'] = liab.sum(-2)

        list_of_graphs.append(g_2)

    return list_of_graphs


def my_loss(graph, eis_noe_clearer, bailout_ratio, bailout_capital):
    cv = get_clearing_vector_iter(graph, bailout_ratio * bailout_capital)# clearer already knows bailout_capital
    out_liab = graph.ndata['outgoingLiabilities']
    loss = (out_liab - cv).sum()
    return loss


def evaluate(
        model: nn.Module,
        bailout_capital,
        list_of_graphs
) -> float:
    model.eval()
    with torch.no_grad():
        # en_eva = EisenbergNoeFast(1.0, 1.0, bailout_capital)
        graph_losses = []
        for i, graph in enumerate(list_of_graphs):
            bailout_ratio = model.predict(graph)
            loss = my_loss(graph=graph,
                           eis_noe_clearer=None,  # en_eva,
                           bailout_ratio=bailout_ratio,
                           bailout_capital=bailout_capital)

            graph_losses.append(loss)

        mean_performance = torch.mean(torch.stack(graph_losses, dim=0)).item()
    return mean_performance


def train(model,
          bailout_capital,
          number_of_epochs,
          list_of_both_graphs,
          seed=None,
          learning_rate=0.01):
    if seed is not None:
        torch.manual_seed(seed)

    start = time.time()
    model.train()
    bailout_capital = torch.Tensor([bailout_capital])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_model_performance = np.inf
    # best_model = model

    performance = evaluate(model, torch.Tensor([0.]), list_of_both_graphs)
    my_print(
        "Epoch {:05d} | {:10.4f} | Capital "
        " {:6.2f} | Time {:6}".format(0, performance, 0, str(timedelta(seconds=round(time.time()-start)))))

    for epoch in range(1, number_of_epochs + 1):
        graph_losses = []
        for i, graph in enumerate(list_of_both_graphs):
            model.train()
            bailout_ratio = model.predict(graph)
            loss = my_loss(graph=graph,
                           eis_noe_clearer=None,
                           bailout_ratio=bailout_ratio,
                           bailout_capital=bailout_capital)

            graph_losses.append(loss)
            optimizer.zero_grad()
            try:
                loss.backward()
                optimizer.step()
            except:
                pass

        if epoch % 10 == 0:
            performance = evaluate(model, bailout_capital, list_of_both_graphs)
            if performance < best_model_performance:
                best_model_performance = performance

            my_print(
                "Epoch {:05d} | {:10.4f} | Capital "
                " {:6.2f} | Time {:7}".format(epoch, performance,
                                   bailout_capital.item(), str(timedelta(seconds=round(time.time()-start)))))  # math.ceil((time.time()-start)/60)))

    my_print(
        "Best Model  | {:10.4f} | Capital "
        " {:6.2f} | Time {:7}".format(best_model_performance,
                           bailout_capital.item(), str(timedelta(seconds=round(time.time()-start)))))  # math.ceil((time.time()-start)/60) ))

    return 'log'

def run_experiment(N,
                   nn_specs,
                   bailout_capital,
                   number_of_epochs,
                   seed,
                   learning_rate):
    list_of_graphs = generate_data(N)

    if nn_specs == 'XPENN':
        model = XPENN()
    elif nn_specs == 'NNL':
        model = NNL(N)
    else:
        model = GNN()

    my_print(f'N: {N}')
    my_print(f'Model: {nn_specs}')
    my_print(f'Capital: {bailout_capital}')
    my_print(f'Epochs: {number_of_epochs}')
    my_print(f'Seed: {seed}')
    my_print(f'Learning Rate: {learning_rate}')

    run_log = train(model, bailout_capital, number_of_epochs, list_of_graphs, seed, learning_rate)

    my_print('')
    return



if __name__ == '__main__':
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    seed = np.random.randint(0, 3000)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_experiment(10, 'GNN', 9, 1000, seed, 0.001)
    run_experiment(10, 'XPENN', 9, 1000, seed, 0.001)
    run_experiment(10, 'NNL', 9, 1000, seed, 0.001)
    run_experiment(20, 'GNN', 19, 1000, seed, 0.001)
    run_experiment(20, 'XPENN', 19, 1000, seed, 0.001)
    run_experiment(20, 'NNL', 19, 1000, seed, 0.0005)
    run_experiment(50, 'GNN', 49, 1000, seed, 0.001)
    run_experiment(50, 'XPENN', 49, 1000, seed, 0.001)
    run_experiment(50, 'NNL', 49, 1000, seed, 0.0001)
    run_experiment(100, 'GNN', 99, 1000, seed, 0.001)
    run_experiment(100, 'XPENN', 99, 1000, seed, 0.001)
    run_experiment(100, 'NNL', 99, 1000, seed, 0.00001)

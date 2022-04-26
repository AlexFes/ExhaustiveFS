import sys
import json
import os
import networkx as nx
import pandas as pd
import numpy as np
import csv
from multiprocessing import Pool
from scipy.stats import spearmanr, pearsonr
from itertools import combinations

def get_knet(G, k):
    adj_matrix = nx.to_numpy_array(G.to_undirected())
    E = np.diag(np.ones(len(G)))
    adj_matrix += E
    adj_matrix_power = np.linalg.matrix_power(adj_matrix, k)
    adj_matrix_power_bin = adj_matrix_power.astype(bool).astype(int)

    derived_graph = nx.from_numpy_array(adj_matrix_power_bin)
    derived_graph_labels = dict(zip(range(len(G.nodes)), G.nodes))
    nx.relabel_nodes(derived_graph, derived_graph_labels, copy=False)

    return nx.algorithms.approximation.min_weighted_dominating_set(derived_graph)


def read_graph(config_path):
    try:
        config_file = open(config_path, "r")
    except:
        print("Cannot open configuration file", file=sys.stderr)
        sys.exit(1)

    try:
        config = json.load(config_file)
    except:
        print("Please provide a valid configuration file", file=sys.stderr)
        sys.exit(1)

    config_dirname = os.path.dirname(config_path)
    df = pd.read_csv(os.path.join(config_dirname, config["data_path"]).replace("\\", "/"), index_col=0)
    ann = pd.read_csv(os.path.join(config_dirname, config["annotation_path"]).replace("\\", "/"), index_col=0)

    try:
        G = nx.read_graphml(config["network_path"])
    except:
        print("Please provide a valid graphml file", file=sys.stderr)
        sys.exit(1)

    return G, df, ann, config


if __name__ == "__main__":
    config_path = "graph_config.json"

    G, df, ann, config = read_graph(config_path)

    if config["build_graph"]:
        datasets = np.unique(ann.loc[ann["Dataset type"] != "Validation", "Dataset"])
        samples = ann.loc[ann["Dataset"].isin(datasets)].index
        data_subset = df.loc[samples]

        feature_pairs = list(combinations(df.columns, 2))
        pairs_count = len(feature_pairs)
        batch_size = pairs_count / config["n_processes"]
        pairs_batch = []
        for i in range(config["n_processes"]):
            if i < config["n_processes"] - 1:
                #pairs_batch.append(feature_pairs[i * batch_size:(i + 1) * batch_size])
                print((i * batch_size,(i + 1) * batch_size))
            else:
                #pairs_batch.append(feature_pairs[i * batch_size:(i + 1) * batch_size + pairs_count % config["n_processes"]])
                print((i * batch_size, (i + 1) * batch_size + pairs_count % config["n_processes"]))


        # correlations_array = []
        # for i in range(pairs_count):
        #     current = spearmanr(df[feature_pairs[i][0]], df[feature_pairs[i][1]])
        #     correlations_array.append(current)
        #     if i % 10000 == 0:
        #         print("Done {} out of {}\n".format(i, pairs_count))
        #         print(current)

    else:
        nodes_to_remove = [node for node in G.nodes if not node in set(df.columns)]
        G.remove_nodes_from(nodes_to_remove)

        # largest_component_nodes = max(nx.strongly_connected_components(G), key=len)
        # largest_component = G.subgraph(largest_component_nodes)

        largest_weak_component_nodes = max(nx.weakly_connected_components(G), key=len)
        largest_weak_component = G.subgraph(largest_weak_component_nodes)

        # print([len(get_knet(largest_weak_component, k)) for k in range(8)])

        top_in_degree = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:25]
        top_out_degree = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:25]

        with open("features.csv", "w") as f:
            write = csv.writer(f)
            write.writerows([get_knet(largest_weak_component, 4),
                             [node_with_degree[0] for node_with_degree in top_in_degree],
                             [node_with_degree[0] for node_with_degree in top_out_degree]])

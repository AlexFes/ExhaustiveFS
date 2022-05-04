import sys
import json
import os
import networkx as nx
import pandas as pd
import numpy as np
import csv
from scipy.stats import spearmanr
from itertools import combinations
import statsmodels.api as sm


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


def read_data(config_path):
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

    return df, ann, config


def read_graph(config_path):
    try:
        G = nx.read_graphml(config["network_path"])
    except:
        print("Please provide a valid graphml file", file=sys.stderr)
        sys.exit(1)

    return G

if __name__ == "__main__":
    config_path = "graph_config.json"

    df, ann, config = read_data(config_path)

    if config["build_graph"]:
        datasets = np.unique(ann.loc[ann["Dataset type"] != "Validation", "Dataset"])
        samples = ann.loc[ann["Dataset"].isin(datasets)].index
        data_subset = df.loc[samples]

        pairs = list(combinations(df.columns, 2))
        pairs_count = len(pairs)
        correlations_array = []
        count = 0

        for pair in pairs:
            correlations_array.append(spearmanr(data_subset[pair[0]], data_subset[pair[1]]))
            count += 1
            if count % 1e6 == 0:
                print("Done {} out of {}\n".format(count, pairs_count))

        reject, p_vals, _, _ = sm.stats.multipletests(pvals=[p_val for _, p_val in correlations_array], alpha=0.05, method='fdr_bh')

        edges = [edge for edge, corr, reject, p_val_corrected in zip(pairs, correlations_array, reject, p_vals) if abs(corr[0]) >= config["correlation_threshold"] and reject and p_val_corrected <= config["p_value_threshold"]]
        G = nx.from_edgelist(edges)
        nx.write_graphml(G, "correlation_graph.graphml")

    else:
        G = read_graph(config_path)
        nodes_to_remove = [node for node in G.nodes if not node in set(df.columns)]
        G.remove_nodes_from(nodes_to_remove)

        # largest_component_nodes = max(nx.strongly_connected_components(G), key=len)
        # largest_component = G.subgraph(largest_component_nodes)

        largest_weak_component_nodes = max(nx.weakly_connected_components(G), key=len)
        largest_weak_component = G.subgraph(largest_weak_component_nodes)

        # print([len(get_knet(largest_weak_component, k)) for k in range(8)])

        top_in_degree = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:15]
        top_out_degree = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:15]

        with open("features.csv", "w") as f:
            write = csv.writer(f)
            write.writerows([get_knet(largest_weak_component, 4),
                             [node_with_degree[0] for node_with_degree in top_in_degree],
                             [node_with_degree[0] for node_with_degree in top_out_degree]])

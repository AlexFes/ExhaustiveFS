import sys
import json
import os
import networkx as nx
import pandas as pd
import numpy as np


def get_k_net(G, k):
    adj_matrix = nx.to_numpy_array(G)
    E = np.diag(np.ones(largest_component_size))
    adj_matrix += E
    adj_matrix_power = np.linalg.matrix_power(adj_matrix, k)
    adj_matrix_power_bin = adj_matrix_power.astype(bool).astype(int) - E.astype(int)

    derived_graph = nx.from_numpy_array(adj_matrix_power_bin, create_using=nx.DiGraph)
    derived_graph_labels = dict(zip(range(len(G.nodes)), G.nodes))
    nx.relabel_nodes(derived_graph, derived_graph_labels, copy=False)

    return nx.algorithms.approximation.min_weighted_vertex_cover(derived_graph)


if __name__ == "__main__":
    config_path = "graph_config.json"

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

    # remove redundant nodes and edges
    nodes_to_remove = [node for node in G.nodes if not node in set(df.columns)]
    G.remove_nodes_from(nodes_to_remove)

    # get largest strongly connected component
    largest_component_nodes = max(nx.strongly_connected_components(G), key=len)
    largest_component = G.subgraph(largest_component_nodes)
    largest_component_size = len(largest_component)

    # get k-net of the largest component
    k_net = get_k_net(largest_component, 2)
    print(k_net)

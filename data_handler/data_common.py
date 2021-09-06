import pandas as pd
import networkx as nx
import numpy as np


def get_graph(inputgraph, delimiter):
    edges = np.loadtxt(inputgraph, delimiter=delimiter, dtype=int)
    graph = nx.Graph()
    graph.add_edges_from(edges[:, :2])
    # graph.remove_edges_from(graph.selfloop_edges())
    # graph = nx.convert_node_labels_to_integers(graph)
    return graph


def make_zero_incremental(arr):
    """
    :param arr: a vector
    :return: zero incremental vector
    """
    count = len(np.unique(arr))
    min = np.min(arr)
    max = np.max(arr)
    if min == 0 and max == (count-1):
        return arr
    elif min == 1 and max == count:
        return arr - 1
    else:
        unique_values = np.unique(arr)
        sorted_values = np.sort(unique_values)
        for idx, val in enumerate(sorted_values):
            arr[arr == val] = idx
        return arr


def make_zero_incremental_list(arr_list):
    """
    :param arr: a vector
    :return: zero incremental vector
    """
    arr = list()
    for ar in arr_list:
        for i in ar:
            arr.append(i)

    unique_values = np.unique(arr)
    sorted_values = np.sort(unique_values)
    for idx, val in enumerate(sorted_values):
        for ar in arr_list:
            ar[ar == val] = idx
    return arr_list


def make_zero_incremental_list_with_mapping(series_list):
    """
    :param arr: a vector
    :return: zero incremental vector
    """
    arr = pd.concat(series_list)

    unique_values = np.unique(arr)
    # sorted_values = np.sort(unique_values)
    d = dict()
    for idx, val in enumerate(unique_values):
        d[val] = idx
    new_list = list()
    for ar in series_list:
        new_list.append(ar.map(d))
    return new_list, d

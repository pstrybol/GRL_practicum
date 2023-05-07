import pandas as pd
import numpy as np
from NetworkAnalysis.InteractionNetwork import DirectedInteractionNetwork
import time


def test_createFullyConnectedNetwork():

    node_names = ['G1', 'G2', 'G3']
    fc_net = DirectedInteractionNetwork.createFullyConnectedNetwork(node_names)

    assert fc_net.interactions.shape[0] == 6
    assert set(list(fc_net.node_names)) == set(node_names)


def test_getInteractionNamed(network_chain_4_directed):

    df = network_chain_4_directed.getInteractionNamed()
    df2 = network_chain_4_directed.getInteractionNamed(return_both_directions=True)
    # Note that that the latter option is strictly speaking useless

    assert 2 * df.shape[0] == df2.shape[0]
    assert 2 * len(set(zip(df.Gene_A, df.Gene_B))) == len(set(zip(df2.Gene_A, df2.Gene_B)))


def test_mergeNetworks():
    #TODO improve asserts
    node_names = ['G1', 'G2', 'G3']
    fc_net = DirectedInteractionNetwork.createFullyConnectedNetwork(node_names)

    node_names2 = ['G1', 'G4', 'G5']

    fc_net2 = DirectedInteractionNetwork.createFullyConnectedNetwork(node_names2)

    total_net = fc_net.mergeNetworks(fc_net2)

    print(total_net)

    assert set(total_net.node_names) == set(node_names + node_names2)
    assert total_net.interactions.shape[0] == 12


def test_isConnected(disconnected_network_directed):
    node_names = ['G1', 'G2', 'G3']
    fc_net = DirectedInteractionNetwork.createFullyConnectedNetwork(node_names)

    assert fc_net.isConnected

    print(disconnected_network_directed)
    print(disconnected_network_directed.isConnected)
    assert not disconnected_network_directed.isConnected


def test_getComponents(disconnected_network_directed):
    comps = disconnected_network_directed.getComponents(return_subgraphs=True)

    assert len(comps) == 3
    assert max(comps, key=len).interactions.shape[0] == 6

    comps_df = disconnected_network_directed.getComponents(return_subgraphs=False)

    assert len(np.unique(comps_df.Component.values)) == 3
    assert comps_df.shape[0] == 9

    fc_net = DirectedInteractionNetwork.createFullyConnectedNetwork(['G1', 'G2', 'G3'])

    comps = fc_net.getComponents(return_subgraphs=True)

    assert len(comps) == 1
    assert max(comps, key=len).interactions.shape[0] == 6


def test_keepLargestComponent(disconnected_network):

    largest_comp = disconnected_network.keepLargestComponent()

    assert largest_comp.N_nodes == 3
    assert largest_comp.interactions.shape[0] == 3

    disconnected_network.keepLargestComponent(inplace=True)

    assert disconnected_network.N_nodes == 3
    assert disconnected_network.interactions.shape[0] == 3


def test_getAdjMatrix():

    #TODO: insert other test scenarios
    fc_net = DirectedInteractionNetwork.createFullyConnectedNetwork(['G1', 'G2', 'G3'])

    A, node_names = fc_net.getAdjMatrix(as_df=False)
    print(A)
    print(A[np.triu_indices(3, k=1)])
    assert np.all(A[np.triu_indices(3, k=1)] == 1)
    assert set(list(node_names)) == {'G1', 'G2', 'G3'}



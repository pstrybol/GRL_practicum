import pandas as pd
import numpy as np
from NetworkAnalysis.InteractionNetwork import UndirectedInteractionNetwork, checkTrainingSetsPairs, Graph
from NetworkAnalysis.DiscreteOmicsDataSet import DiscreteOmicsDataSet
import time

test_interaction_df = pd.DataFrame({'Column1': ['Maarten', 'Delphine', 'Michiel', 'Celine'],
                                    'Column2': ['Michiel', 'Maarten', 'Delphine', 'Joachim']})

test_undir = UndirectedInteractionNetwork(test_interaction_df)
test_graph = Graph(test_interaction_df)

print(test_graph.node_names)
print(test_graph.nodes)

print(test_undir.node_names)
print(test_undir.nodes)

# the UndirectedInteractionNetwork is basically a pandas Dataframe, where genes are internally represented as integers
print(test_undir.interactions)

# To work on direcly on the dataframe with the genenames we can use:
print(test_undir.getInteractionNamed())

# If we want to make a copy of an object, it is safest to use the deepcopy command (see below)
test_copy = test_undir.deepcopy()

# First thing we do is to check whether the graph is connected
components, connected = test_copy.getComponents()
print(connected)

comps = test_copy.getComponents(return_subgraphs=True)

# We see that the graph is disconnected, if we want to only keep the largest component we can do:
test_copy.keepLargestComponent(inplace=False).getInteractionNamed()

# Note that this can also be done during initializiation, by specifying:
# test_undir = UndirectedInteractionNetwork(test_interaction_df, keeplargestcomponent=True)

# The nodes can be mapped onto new nodes, by providing a dictionary containing old-new pairs
# incomplete mapping, performs the mapping but returns a warning.
incomplete_dict = {'Maarten': 'Gene A', 'Delphine': 'Gene B', 'Michiel': 'Gene C', 'Celine': 'Gene Z'}
test_copy.mapNodeNames(incomplete_dict)

print(test_copy.node_names)

# Complete mapping
test_copy = test_undir.deepcopy()

complete_dict = {'Maarten': 'Gene A', 'Delphine': 'Gene B', 'Michiel': 'Gene C',
                 'Celine': 'Gene Z', 'Joachim': 'Gene ZZ'}

test_copy.mapNodeNames(complete_dict)
print(test_copy.node_names)
print(test_undir.node_names)

# Note that if we don't use deepcopy we can have the following behaviour:
test_copy = test_undir.deepcopy()
test_copy2 = test_copy

test_copy.mapNodeNames(complete_dict)
print(test_copy.node_names)
print(test_copy2.node_names)
# test_copy2 is also changed as both test_copy and test_copy2 are references to the same memory block

# new interaction information can easily be included in the interaction information
test_copy = test_undir.deepcopy()
new_int = pd.DataFrame({'Gene_A': 'Maarten', 'Gene_B': 'Celine'}, index=[0])

test_copy = test_copy.mergedf(new_int) # merge with dataframe
test_undir = test_undir.mergeNetworks(UndirectedInteractionNetwork(new_int)) #merge with other network object

# To remove nodes we can either provide a list of the genes we want to keep
test_undir.subsetNetwork(['Maarten', 'Celine', 'Joachim']).getInteractionNamed()
test_copy.subsetNetwork(['Maarten', 'Celine', 'Joachim'], inplace=True)
test_copy.getInteractionNamed()

# Or a list of genes that are to be removed
test_undir.removeNodes(['Delphine', 'Michiel'], inplace=False).getInteractionNamed()

# Additionally, we can prune the network,i.e. remove the leave nodes:
pruned_test = test_undir.pruneNetwork()  # exception_list=['Maarten', 'Celine'])
pruned_test.getInteractionNamed()
test_undir.getInteractionNamed()

test_copy = test_undir.deepcopy()
test_copy.pruneNetwork(inplace=True)
test_copy.getInteractionNamed()

# If we want to map a list of genes on the network we can use the following:
gene_list = ['Noise', 'Maarten', 'Celine', 'Michiel']
test_undir.checkInteraction_list(gene_list)

# If we want to know for a dataframe of gene pairs, which interaction are present, we can use:
random_interaction_df = pd.DataFrame({'Column1': ['Maarten', 'Delphine', 'Maarten', 'Celine', 'Noise'],
                                    'Column2': ['Michiel', 'Maarten', 'Delphine', 'Michiel', 'Noise']})

int_df = test_undir.checkInteractions_df(random_interaction_df, colnames=('Column1', 'Column2'))

# We can easily obtain the adjacency class and the nodes
test_undir.getAdjDict()
A, nodes = test_undir.getAdjMatrix()

# The PyLouvain algorithm can be used to cluster nodes from the graph
test_undir.findcommunities(verbose=True)

# We can easily find the Geodesic distance (=shortest path distance between two genes) between two lists of genes
test_undir.getGeodesicDistance(test_undir.node_names, test_undir.node_names)

# Similarly, we can find the nth order neighbors using:
test_undir.getNOrderNeighbors(order=2, include_lower_order=False)
# Warning: This method becomes very slow on large networks
# If we only want to know this for some genes, we can run:
test_undir.getNOrderNeighbors(order=2, include_lower_order=True, gene_list=['Maarten'])

# Find a minimum spanning tree
test_undir.getMinimmumSpanningTree()

# If we want to apply diffusion on the adjacency matrix:
test_lex = test_undir.diffuse(kernel='LEX', alpha=0.15)
test_rwr = test_undir.diffuse(kernel='RWR', alpha=0.15)


# As a convenience function, we can direcly use agglomerative clustering on the network object
test_undir.clusterbydiffusion(kernel='LEX', alpha=0.01, nclusters=2, linkage='average', verbose=True)

# The network class can be direcly used to generate a training and testing set,
#  consisting of positive and negative pairs, a summary of the training and test set can be obtained can be obtained
X_train, X_test = test_undir.getAllTrainData()
X_train_ms, X_test_ms, Y_train_ms, Y_test_ms, summary_ms_tree = test_undir.getTrainTestData(method='ms_tree')
X_train_bl, X_test_bl, Y_train_bl, Y_test_bl, summary_balanced = test_undir.getTrainTestData(method='balanced')

# to run some additional checks on the data we use the checkTrainingSetsPairs function
checkTrainingSetsPairs(X_train_ms, Y_train_ms, X_test_ms, Y_test_ms)
checkTrainingSetsPairs(X_train_bl, Y_train_bl, X_test_bl, Y_test_bl)


# The Interaction network can be filtered together with dataset objects:
testdataset_mut = DiscreteOmicsDataSet(pd.DataFrame(np.array([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]),
                                                    index=['P1', 'P2', 'P3'],
                                                    columns=['Michiel', 'Delphine', 'Maarten', 'Pieter', 'Kristin']), type='MUT')
# If we want to filter the dataset to only keep the genes in the network we can use
filtered_net, dataset_filtered = test_undir.filterDataset(testdataset_mut, inplace=False)
print(dataset_filtered.df)

# Optionally, if one wants to prune the dataset, in such a way that the leaves for which we have data are still kept:
filtered_net_pruned, dataset_filtered = test_undir.filterDataset(testdataset_mut, inplace=False, remove_leaves=True)
print(dataset_filtered.df)
print(filtered_net_pruned.getInteractionNamed())

testdataset_expr = DiscreteOmicsDataSet(pd.DataFrame(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                                     index=['P1', 'P2', 'P3'],
                                                     columns=['Michiel', 'Delphine', 'Maarten']), type='EXPR')

# In case there are several datasets, we can filter all of these networks at once
test_undir.filterDatasetGenes([testdataset_mut, testdataset_expr])


# let's check on a real interaction network:
net = pd.read_csv('/home/mlarmuse/Documents/NetworkData/kegg-acsn-regnetwork.csv')
graph = Graph(net[['node1', 'node2']])
net = UndirectedInteractionNetwork(net[['node1', 'node2']])


small_comps = [subnet.node_names for subnet in net.getComponents(return_subgraphs=True) if len(subnet) < 3]
largest_comp = net.keepLargestComponent(inplace=False)

print([subnet.node_names for subnet in net.findcommunities() if 'CD70' in subnet])

subnets = net.keepLargestComponent().findcommunities()
subnet_df = net.keepLargestComponent().findcommunities(as_df=True)

for subnet in subnets:
    if len(subnet) < 100:
        subnet.visualize(show_labels=True)

from NetworkAnalysis.PyLouvain import PyLouvain, in_order

df = net.getInteractionNamed()
nodes = {node: 1 for node in net.node_names}
edges = [(pair, 1) for pair in zip(df.Gene_A, df.Gene_B)]
nodes_, edges_, map_dict = in_order(nodes, edges, return_dict=True)
PyL_object = PyLouvain(nodes_, edges_)
PyL_object.apply_method()

PyL_object.actual_partition



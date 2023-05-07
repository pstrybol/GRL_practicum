import pandas as pd
import numpy as np
import networkx as nx
from networkx import to_dict_of_lists
from NetworkAnalysis.Graph import Graph, positives_split

# complete the class specific functions in Directed networks


class DirectedInteractionNetwork(Graph):
    def __init__(self, interaction_df, colnames=None, verbose=True, keeplargestcomponent=False,
                 allow_self_connected=False, node_types=None, gene2int=None):
        super().__init__(interaction_df, colnames, verbose=verbose, keeplargestcomponent=keeplargestcomponent,
                         allow_self_connected=allow_self_connected, node_types=node_types, gene2int=gene2int)
        self.directed = True

    @classmethod
    def createFullyConnectedNetwork(cls, node_names):
        df = pd.DataFrame(np.array([(n1, n2) for n1 in node_names
                                    for n2 in node_names]),
                          columns=['gene_A', 'Gene_B'])

        return cls(df)

    @property
    def isConnected(self):
        G = self.getnxGraph()
        return nx.is_weakly_connected(G)

    def mergedf(self, interaction_df, colnames=None):
        return self.mergeNetworks(DirectedInteractionNetwork(interaction_df,
                                                             colnames=colnames,
                                                             verbose=False,
                                                             node_types=self.node_type_names))

    def mergeNetworks(self, network):
        new_dict = {**self.node_type_names, **network.node_type_names}
        return DirectedInteractionNetwork(pd.concat([self.getInteractionNamed(), network.getInteractionNamed()],
                                                    ignore_index=True,
                                                    axis=0),
                                          node_types=new_dict)

    '''
        get adj_dict
    '''

    def getAdjDict(self, return_names=True):
        return to_dict_of_lists(self.getnxGraph(return_names=return_names))

    def getnxGraph(self, return_names=True):
        '''return a graph instance of the networkx module'''
        if return_names:
            df = self.getInteractionNamed()
        else:
            df = self.interactions

        return nx.from_pandas_edgelist(df, source='Gene_A', target='Gene_B', create_using=nx.DiGraph)

    def getGeodesicDistance(self, start_genes, stop_genes, nx_Graph=None):
        '''
        :param: start_genes genes from which to find paths to stop_genes
        :return: a pandas df containing the pathlengths of shape (start_genes, stop_genes)
        '''

        if isinstance(start_genes[0], str):
            node_names = self.node_names
            start_genes_ = np.intersect1d(node_names, start_genes)
            stop_genes_ = np.intersect1d(node_names, stop_genes)
            return_names = True

        else:
            start_genes_ = np.intersect1d(self.nodes, start_genes)
            stop_genes_ = np.intersect1d(self.nodes, stop_genes)
            return_names = False

        if len(start_genes_) == 0:
            raise IOError('The start_genes are not known.')

        if nx_Graph is None:
            nx_Graph = self.getnxGraph(return_names=return_names)

        path_lengths = np.zeros((len(start_genes_), len(stop_genes_)))

        for istop, stop in enumerate(stop_genes_):
            for istart, start in enumerate(start_genes_):

                try:
                    path_lengths[istart, istop] = len(nx.shortest_path(nx_Graph, start, stop)) - 1

                except nx.NetworkXNoPath:  # basically the graph is not fully connected
                    path_lengths[istart, istop] = np.nan

        # paths_lengths = np.array([len(nx.shortest_path(A, start, stop)) - 1 for stop in stop_genes for start in start_genes])

        paths_lengths_df = pd.DataFrame(path_lengths, index=start_genes_, columns=stop_genes_)
        return paths_lengths_df

    def findAllTrees(self):
        '''
        :return: A list of all maximum trees, one for each root
        '''
        pass

    def subsample(self, n=100, weighted=False):
        return DirectedInteractionNetwork(super().subsample(n=n, weighted=weighted),
                                          node_types=self.node_type_names)

    def makeSelfConnected(self, inplace=False):
        if not inplace:
            return DirectedInteractionNetwork(super(DirectedInteractionNetwork, self).
                                              makeSelfConnected(inplace=False), colnames=('Gene_A', 'Gene_B'),
                                              allow_self_connected=True,
                                              node_types=self.node_type_names,
                                              gene2int=self.gene2int)

    def removeNodes(self, nodes_tbr, inplace=False):

        if inplace:
            super().removeNodes(nodes_tbr, inplace=inplace)
        else:
            return DirectedInteractionNetwork(super().removeNodes(nodes_tbr, inplace=False),
                                              node_types=self.node_type_names)

    def subsetNetwork(self, nodes, inplace=False, verbose=True):

        if inplace:
            self.setEqual(DirectedInteractionNetwork(super().subsetNetwork(nodes),
                                                     node_types=self.node_type_names))

        else:
            return DirectedInteractionNetwork(super().subsetNetwork(nodes),
                                              verbose=verbose,
                                              allow_self_connected=True,
                                              node_types=self.node_type_names)

    def getMinimmumSpanningTree(self, as_edge_list=True, return_names=True):

        edge_list = super().getMinimmumSpanningTree(return_names=return_names)

        if as_edge_list:
            return edge_list

        else:
            edges = np.array(edge_list)
            df = pd.DataFrame(edges, columns=["Gene_A", "Gene_B"])
            return DirectedInteractionNetwork(df, node_types=self.node_type_names)

    def getDegreeDF(self, return_names=True, degree='total-degree'):
        '''
        :param return_names:  whether the node names should be returned.
        :param degree: the definition of degree, option are (total-degree, in-degree, out-degree).
        default: total-degree
        :return: dgree DF
        '''

        if degree.lower() == 'in-degree':
            v, c = np.unique(self.interactions['Gene_B'].values, return_counts=True)

        elif degree.lower() == 'out-degree':
            v, c = np.unique(self.interactions['Gene_A'].values, return_counts=True)

        else:
            v, c = np.unique(self.interactions.values, return_counts=True)

        if return_names:
            return pd.DataFrame({'Gene': [self.int2gene[i] for i in v],
                                 'Count': c}).sort_values(by='Count', ascending=False, inplace=False)
        else:
            return pd.DataFrame({'Gene': v,
                                 'Count': c}).sort_values(by='Count', ascending=False, inplace=False)

    def getComponents(self, return_subgraphs=False, verbose=False):
        '''
        Function that returns the connected components of a graph
        :param return_subgraphs: whether the subgraphs need to be returned as a list of UndirectedInteractionNetwork instances
        or as a pandas DF with for each node a label.
        :param verbose:
        :return: Either a pandas DF with the node labels denoting the different components or a list of
        UndirectedNetwork instances
        '''

        node_names = self.node_names
        if self.isConnected:

            if return_subgraphs:
                return [self.deepcopy()]

            else:
                return pd.DataFrame({'Gene': node_names, 'Component': [0 for _ in node_names]})

        else:
            components = nx.weakly_connected_components(self.getnxGraph())

            if not return_subgraphs:
                map_dict = {node: i for i, subgraph_nodes in enumerate(components) for node in subgraph_nodes}
                return pd.DataFrame({'Gene': node_names, 'Component': [map_dict[x] for x in node_names]})

            else:
                return [self.subsetNetwork(subgraph, verbose=verbose) for subgraph in components]

    def replaceNodesWithInteractions(self, tbr_nodes):
        df = super().replaceNodesWithInteractions(tbr_nodes)
        return DirectedInteractionNetwork(df, node_types=self.node_type_names)

    def getTrainTestPairs_MStree(self, train_ratio=0.7, train_validation_ratio=0.7, excluded_sets=None,
                                 neg_pos_ratio=5, check_training_set=False, random_state=42):
        '''
        :param: train_ratio: The fraction of samples used for training
        :param: neg_pos_ratio: The ratio of negative examples to positive examples
        :param: assumption: Whether we work in the open world or  closed world assumption
        :return: positive and negative pairs for both train and test set (4 lists in total)
        '''
        # TODO: update this function for validation split
        np.random.seed(random_state)
        # To get the negatives we first build the adjacency matrix
        df = self.interactions
        df.values.sort(axis=1)

        allpos_pairs = set(zip(df.Gene_A, df.Gene_B))
        pos_train, pos_valid, pos_test = positives_split(df, train_ratio)

        N_neg = np.int(neg_pos_ratio * len(allpos_pairs))
        margin = np.int(0.3 * N_neg)

        row_c = np.random.choice(self.N_nodes, N_neg + margin, replace=True)
        col_c = np.random.choice(self.N_nodes, N_neg + margin, replace=True)

        all_pairs = set([(r_, c_) for r_, c_ in zip(row_c, col_c) if (c_ != r_)])

        all_neg = np.array(list(all_pairs.difference(allpos_pairs)), dtype=np.uint16)

        if len(all_neg) > N_neg:
            all_neg = all_neg[:N_neg]
        elif len(all_neg) < N_neg:
            print('The ratio of negatives to positives is lower than the asked %f.'
                  '\nReal ratio: %f' % (neg_pos_ratio, len(all_neg) / len(allpos_pairs)))

        train_ids = np.int(len(all_neg) * train_ratio)
        neg_train, neg_test = all_neg[:train_ids], all_neg[train_ids:]

        if check_training_set:
            degrees = self.getDegreeDF(return_names=False)
            degrees.index = degrees.Gene.values

            genes, counts = np.unique(all_neg.flatten(), return_counts=True)
            df = pd.DataFrame({'Gene': [self.int2gene[g] for g in genes], 'Counts': counts,
                               'Expected': degrees['Count'].loc[genes].values * neg_pos_ratio})
            df['Difference'] = df.Expected - df.Counts
            return list(pos_train), list(neg_train), list(pos_test), list(neg_test), df

        else:
            return list(pos_train), list(neg_train), list(pos_test), list(neg_test)

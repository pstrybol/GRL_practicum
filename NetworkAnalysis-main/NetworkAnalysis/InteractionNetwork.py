from NetworkAnalysis.UndirectedInteractionNetwork import UndirectedInteractionNetwork
from networkx.exception import NetworkXError, AmbiguousSolution
from scipy.linalg import expm
from networkx import to_dict_of_lists
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import warnings
import copy
import time

# complete the class specific functions in Directed networks


class Graph:
    '''
        Builds a graph from _path.
        _path: a path to a file containing "node_from node_to" edges (one per line)
    '''

    @classmethod
    def from_file(cls, path, colnames=('Gene1', 'Gene2'), sep=',',
                  header=0, column_index=None, keeplargestcomponent=False,
                  network_type='kegg', gene_id_type='symbol'):

        if network_type is None:
            network_df = pd.read_csv(path, sep=sep, header=header, low_memory=False, index_col=column_index)
            network_df = network_df[list(colnames)]
        elif network_type.lower() == 'kegg':
            network_df = pd.read_csv(path, sep='\t', header=0, dtype=str)[['from', 'to']]

        elif network_type.lower() == 'string':
            network_df = pd.read_csv(path, sep='\t', header=0)[['Gene1', 'Gene2']]

        elif network_type.lower() == 'biogrid':
            network_df = pd.read_csv(path, sep='\t', header=0)

            if gene_id_type.lower() == 'entrez':
                network_df = network_df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']]

            elif gene_id_type.lower() == 'symbol':
                network_df = network_df[['Official Symbol Interactor A', 'Official Symbol Interactor B']]

            else:
                raise IOError('gene_id_type not understood.'
                              'For Biogrid please specify entrez or symbol.')

        else:
            raise IOError('Network type not understood.'
                          'Please specify kegg, biogrid or reactome, or enter None for custom network type.')

        return cls(network_df, keeplargestcomponent=keeplargestcomponent)

    def __init__(self, interaction_df, colnames=None, verbose=True, keeplargestcomponent=False,
                 allow_self_connected=False, node_types=None, drop_duplicates=True,
                 gene2int=None):
        '''
        :param: interaction_df a pandas edgelist consisting of (at least two) columns,
        indicating the two nodes for each edge
        :param: colnames, the names of the columns that contain the nodes and optionally some edge attributes.
        :param: node_types, dictionary with node types = keys, node names = values
        The first two columns must indicate the nodes from the edgelsist
        '''

        def isinteger(x):
            try:
                return np.all(np.equal(np.mod(x, 1), 0))

            except:
                return False

        self.attr_names = None

        if colnames is not None:
            interaction_df = interaction_df[list(colnames)]
            if len(colnames) > 2:
                self.attr_names = colnames[2:]  # TODO this needs to be done better

        elif interaction_df.shape[1] == 2:
            interaction_df = interaction_df

        else:
            print('Continuing with %s and %s as columns for the nodes' % (interaction_df.columns.values[0],
                                                                          interaction_df.columns.values[1]))
            interaction_df = interaction_df.iloc[:, :2]

        if drop_duplicates:
            interaction_df = interaction_df.drop_duplicates()

        self.interactions = interaction_df
        old_col_names = list(self.interactions.columns)
        self.interactions.rename(columns={old_col_names[0]: 'Gene_A',
                                          old_col_names[1]: 'Gene_B'},
                                 inplace=True)

        if not allow_self_connected:
            self.interactions = self.interactions.loc[self.interactions.Gene_A != self.interactions.Gene_B]

        if isinteger(self.interactions.Gene_A.values):  # for integer nodes do numerical ordering of the node_names
            node_names = np.unique(self.interactions[['Gene_A', 'Gene_B']].values)
            self.interactions = self.interactions.astype(str)
            node_names = node_names.astype(str)

        else:
            self.interactions = self.interactions.astype(str)
            node_names = np.unique(self.interactions[['Gene_A', 'Gene_B']].values)

        if gene2int is not None:
            assert isinstance(gene2int, dict), "If provided, gene2int must be a dict mapping nodenames onto ints."
            gene2int = check_node_dict(node_names, gene2int, type_dict="gene2int")
            assert len(gene2int.values()) == len(set(gene2int.values())), "if provided, " \
                                                                          "gene2int must map each nodename onto a unique int."
            self.int2gene = {i: g for g, i in gene2int.items()}

        else:
            self.int2gene = {i: name for i, name in enumerate(node_names)}
            gene2int = self.gene2int

        self.interactions = self.interactions.applymap(lambda x: gene2int[x])
        self.nodes = np.array([gene2int[s] for s in node_names]).astype(np.int)

        if node_types is None:
            self.node_types = {i: "node" for i in self.nodes}

        elif isinstance(node_types, dict):
            node_type_names = check_node_dict(self.node_names, node_types, type_dict="node_types")
            self.node_types = {self.gene2int[k]: v for k, v in node_type_names.items()}

        else:
            raise IOError("The node_types are not understood, "
                          "please provide a dict mapping each node on their node type.")

        self.embedding_dict = None
        if keeplargestcomponent:
            self.keepLargestComponent(verbose=verbose, inplace=True)

        if verbose:
            print('%d Nodes and %d interactions' % (len(self.nodes),
                                                    self.interactions.shape[0]))

    def deepcopy(self):
        return copy.deepcopy(self)

    @property
    def node_type_names(self):
        return {self.int2gene[g]: t for g, t in self.node_types.items()}

    @property
    def gene2int(self):
        return {v: k for k, v in self.int2gene.items()}

    @property
    def node_names(self):
        return np.array([self.int2gene[i] for i in self.nodes])

    @property
    def N_nodes(self):
        return len(self.nodes)

    @property
    def N_interactions(self):
        return self.interactions.shape[0]

    @property
    def type2nodes(self):
        if self.node_types is not None:
            odict = defaultdict(list)
            for k, v in self.node_types.items():
                odict[v].append(k)

        else:
            odict = None

        return odict

    @property
    def _get_edge_ids(self):
        edge_ids = self.interactions["Gene_A"].values * self.N_nodes + self.interactions["Gene_B"].values
        return edge_ids

    @property
    def is_bipartite(self):
        G = self.getnxGraph(return_names=True)
        bipartite = True

        try:
            l, r = nx.bipartite.sets(G.to_undirected())
            return bipartite, l, r

        except (NetworkXError, AmbiguousSolution) as e:
            bipartite = False
            return bipartite, None, None

    def __contains__(self, gene):
        return gene in self.node_names

    def __repr__(self):
        return self.getInteractionNamed().__repr__()

    def __str__(self):
        return self.getInteractionNamed().__str__()

    def __len__(self):
        return self.N_nodes

    def __eq__(self, other):
        if isinstance(other, Graph):
            return self.interactions_as_set() == other.interactions_as_set()
        return NotImplemented

    def edge_list(self, return_names=True):
        if return_names:
            df = self.getInteractionNamed()

        else:
            df = self.interactions

        return list(zip(df.Gene_A.values, df.Gene_B.values))

    def set_node_types(self, node_types):
        if isinstance(node_types, dict):
            node_type_names = check_node_dict(self.node_names, node_types, type_dict="node_types")
            self.node_types = {self.gene2int[k]: v for k, v in node_type_names.items()}

        else:
            raise IOError("The node_types are not understood, "
                          "please provide a dict mapping each node on their node type.")

    def get_node_type_subnet(self, type, inplace=False):
        """
        returns the subnetwork containing all nodes of a particular type
        :param type:
        :param inplace:
        :return:
        """
        try:
            genes = self.type2nodes[type]

        except KeyError:
            raise IOError("The type is not known, please check that the type is present in node_types.")

        if inplace:
            self.removeNodes(genes, inplace=inplace)

        else:
            return self.removeNodes(genes, inplace=inplace)

    def get_interactions_per_node_type(self):
        if self.node_types is None:
            return None

        node_type_interactions = self.interactions[["Gene_A", 'Gene_B']].applymap(lambda x: self.node_types[x])
        uniq = set(node_type_interactions.itertuples(index=False, name=None))

        return uniq

    def get_node_type_edge_counts(self):
        if self.node_types is not None:
            df = self.interactions[["Gene_A", 'Gene_B']].applymap(lambda x: self.node_types[x])
            return df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'counts'})

        else:
            warnings.warn("The node types are not defined, please provide these first.")
            return None

    def get_edges_by_node_types(self, node_type1, node_type2, return_names=True, directed=False):
        df = self.interactions[["Gene_A", 'Gene_B']].applymap(lambda x: self.node_types[x]).values

        if directed:
            mask = (df[:, 0] == node_type1) & (df[:, 1] == node_type2)

        else:
            mask = ((df[:, 0] == node_type1) & (df[:, 1] == node_type2)) | \
                   ((df[:, 0] == node_type2) & (df[:, 1] == node_type1))

        if return_names:
            return self.getInteractionNamed()[mask]
        else:
            return self.interactions[mask]

    def sample_positives_and_negatives(self, neg_pos_ratio=5, excluded_sets=None, return_names=False):
        # TODO: make excluded sets name-based instead of number-based, without breaking getTrainTestPairs_MStree_ML
        directed = isinstance(self, DirectedInteractionNetwork)
        edge_counts = self.get_node_type_edge_counts()
        all_neg, all_pos = [], []
        type2nodes = self.type2nodes

        if excluded_sets is None:
            excluded_sets = set()

        elif not isinstance(self, DirectedInteractionNetwork):
            excluded_sets = set([tuple(sorted((r_, c_))) for r_, c_ in excluded_sets])

        for row in edge_counts.itertuples(index=False, name=None):
            # select all positives from the training set
            positives = self.get_edges_by_node_types(node_type1=row[0], node_type2=row[1], return_names=False,
                                                     directed=directed)

            sets = set(zip(positives.Gene_A, positives.Gene_B))

            positives = set(positives.itertuples(index=False, name=None)).difference(excluded_sets)

            # get negatives based on the count
            N_negatives = np.round(len(positives) * neg_pos_ratio).astype(int)

            nodes1, nodes2 = type2nodes[row[0]], type2nodes[row[1]]

            margin = np.int(0.3 * N_negatives)

            row_c = np.random.choice(nodes1, N_negatives + margin, replace=True)
            col_c = np.random.choice(nodes2, N_negatives + margin, replace=True)

            if isinstance(self, UndirectedInteractionNetwork):
                assert len(sets) == len(set([tuple(sorted(t)) for t in sets])), "Duplicate edges pos"

                all_pairs = set([tuple(sorted((r_, c_))) for r_, c_ in zip(row_c, col_c) if (c_ != r_)]) # should be sorted for Undirected

            else:
                all_pairs = set([(r_, c_) for r_, c_ in zip(row_c, col_c) if (c_ != r_)])

            if excluded_sets is not None:
                negatives = list(all_pairs - positives - set(excluded_sets))

            else:
                negatives = list(all_pairs.difference(positives))

            if len(negatives) > N_negatives:
                negatives = negatives[:N_negatives]

            if len(negatives) < N_negatives:
                print('The ratio of negatives to positives is lower than the asked %f.'
                      '\nReal ratio: %f' % (neg_pos_ratio, len(negatives) / len(positives)))

            excluded_sets = excluded_sets.union(positives).union(set(negatives))

            all_neg += negatives
            all_pos += positives

        if return_names:
            func_ = lambda x: self.int2gene[x]
            # func_reverse_ = lambda x: net.gene2int[x] if x in net.gene2int else np.nan # genes in one might not be in other
            vf = np.vectorize(func_)

            return vf(np.asarray(all_pos)), vf(np.asarray(all_neg))

        else:
            return np.asarray(all_pos), np.asarray(all_neg)

    def update_node_type(self, new_dict):
        for k, v in new_dict:
            self.node_types[k] = v

    def get_type_Series(self):
        return pd.Series(self.node_types)

    def remove_nodes_from_type_dict(self, genes):
        for gene in genes:
            self.node_types.pop(gene, None)

    def check_node_dict(self, node_dict):
        """
        expects a dict where keys = nodes in the network, values are the corresponding types
        """
        net_node_names = self.node_names
        nodes = list(node_dict.keys())

        nodes_missing_in_dict = set(net_node_names) - set(nodes)

        if len(nodes_missing_in_dict) > 0:
            print("The following genes have no annotation:")
            print(nodes_missing_in_dict)
            return None

        nodes_missing_in_network = set(nodes) - set(net_node_names)

        if len(nodes_missing_in_network) > 0:
            print("The following genes are missing from the network and will be removed:")
            print(nodes_missing_in_network)
            return {k: v for k, v in node_dict.items() if k not in nodes_missing_in_network}
        else:
            return node_dict

    def getInteractionNamed(self, return_both_directions=False):
        if return_both_directions:
            df = self.interactions.applymap(lambda x: self.int2gene[x])
            df2 = df.copy(deep=True).rename(columns={'Gene_B': 'Gene_A', 'Gene_A': 'Gene_B'})
            return pd.concat([df, df2], axis=0, ignore_index=True)
        else:
            return self.interactions.applymap(lambda x: self.int2gene[x])

    def interactions_as_set(self, return_names=True):
        if return_names:
            df = self.getInteractionNamed()
        else:
            df = self.interactions
        return set(zip(df.Gene_A.values, df.Gene_B.values))

    def getInteractionInts_as_tuple(self, both_directions=False):
        tup = (self.interactions.Gene_A.values, self.interactions.Gene_B.values)

        if both_directions:
            tup = (np.hstack((tup[0], tup[1])), np.hstack((tup[1], tup[0])))

        return tup

    def setEqual(self, network):
        '''
        Convenience function for setting the attributes of a network equal to another network
        '''
        self.interactions = copy.deepcopy(network.interactions)
        self.nodes = copy.deepcopy(network.nodes)

        self.int2gene = copy.deepcopy(network.int2gene)
        self.attr_names = copy.deepcopy(network.attr_names)

    def mapNodeNames(self, map_dict):
        if len(set(self.node_names).difference(set(list(map_dict.keys())))) > 0:
            warnings.warn('The provided mapping does not convert all ids, for these nodes, old IDs will be kept.')
        
        int2gene = self.int2gene
        self.int2gene = {i: map_dict[name] if (name in map_dict.keys()) else name for i, name in int2gene.items()}

    def subsetNetwork(self, nodes, inplace=True):
        nodes = set(nodes)
        df = self.getInteractionNamed()
        df = df.loc[df.Gene_A.isin(nodes) &
                    df.Gene_B.isin(nodes)]

        return df

    '''
        merge networks
    '''

    def makeSelfConnected(self, inplace=False):
        self_df = pd.DataFrame({'Gene_A': self.node_names, 'Gene_B': self.node_names})

        if inplace:
            self_df = self_df.applymap(lambda x: self.gene2int[x])
            self.interactions = pd.concat([self.interactions, self_df], ignore_index=True)

        else:
            new_df = pd.concat([self.getInteractionNamed(), self_df], ignore_index=True)
            return new_df

    def mergeNetworks(self, network):
        pass

    def mergedf(self, interaction_df, colnames=None):
        pass

    '''
        get adjacency matrix
    '''

    def getAdjMatrix(self, sort='first', as_df=False):

        row_ids = list(self.interactions['Gene_A'])
        col_ids = list(self.interactions['Gene_B'])

        A = np.zeros((self.N_nodes, self.N_nodes), dtype=np.uint8)
        A[(row_ids, col_ids)] = 1

        if as_df:
            return pd.DataFrame(A, index=self.node_names, columns=self.node_names)
        else:
            return A, np.array(self.node_names)

    def normalizeAdjecencyMatrix(self, symmetric_norm=False):
        adj_array, node_names = self.getAdjMatrix()

        if symmetric_norm:
            D = np.diag(1. / np.sqrt(np.sum(adj_array, axis=0)))
            adj_array_norm = np.dot(np.dot(D, adj_array), D)
        else:
            degree = np.sum(adj_array, axis=0)
            adj_array_norm = (adj_array * 1.0 / degree).T

        return pd.DataFrame(adj_array_norm, index=node_names, columns=node_names)

    '''
        perform kernel diffusion
    '''

    def diffuse(self, kernel='LEX', alpha=0.01, as_df=True, scale=False, self_connected=True,
                symmetric_norm=False):

        A, nodes = self.getAdjMatrix()

        if self_connected:
            np.fill_diagonal(A, np.uint8(1))

        A = A.astype(np.float)
        starttime = time.time()
        if kernel.upper() == 'LEX':   # TODO insert other diffusion techniques
            A = np.diag(np.sum(A, axis=0)) - A
            # for undirected graphs the axis does not matter, for directed graphs use the in-degree
            A = expm(-alpha*A)

        elif kernel.upper() == 'RWR':
            term1 = (1 - alpha) * A
            term2 = np.identity(A.shape[1]) - alpha * self.normalizeAdjecencyMatrix(symmetric_norm=symmetric_norm).values
            term2_inv = np.linalg.inv(term2)
            A = np.dot(term1, term2_inv)

        elif kernel.upper() == 'VANDIN':
            A = np.diag(np.sum(A, axis=0)) - A
            # Adjust diagonal of laplacian matrix by small gamma as seen in Vandin 2011
            A = np.linalg.inv(A + alpha * np.identity(self.N_nodes))
            # status: block tested with the original code

        if scale:
            A = A / np.outer(np.sqrt(np.diag(A)), np.sqrt(np.diag(A)))
        print('Network Propagation Complete: %i seconds' %(time.time() - starttime))
        if as_df:
            df = pd.DataFrame(A, index=nodes, columns=nodes)
            return df
        else:
            return A, nodes

    def propagateMutations(self, mut_df, scale_mutations=False, precomputed_kernel=None, **kernelargs):

        if precomputed_kernel is None:
            K = self.diffuse(as_df=True, **kernelargs)
        else:
            assert isinstance(precomputed_kernel, pd.DataFrame), "Please provide the mutation data as a pandas DataFrame."
            K = precomputed_kernel

        assert isinstance(mut_df, pd.DataFrame), "Please provide the mutation data as a pandas DataFrame."

        mut_genes = mut_df.columns.values
        K_genes = K.columns.values

        common_genes = np.intersect1d(K_genes, mut_genes)

        if len(common_genes) == 0:
            mut_df = mut_df.transpose()
            mut_genes = mut_df.columns.values

            common_genes = np.intersect1d(K_genes, mut_genes)

            if len(common_genes) == 0:
                raise IOError('There are not genes in common between the mutation dataframe and the network kernel.')

        print('There are %i genes in common between the mutation dataset and the network.' % len(common_genes))
        mut_df = mut_df[common_genes]
        
        if scale_mutations:
            mut_df = mut_df / np.sum(mut_df.values, keepdims=True, axis=1)
        
        diff_scores = np.matmul(mut_df.values, K.loc[common_genes].values)

        return pd.DataFrame(diff_scores, index=mut_df.index, columns=K.columns.values)

    '''
        check interactions given a list
        find all interactions between the genes in a list
    '''

    def checkInteraction_list(self, gene_list, attribute_separator=None):

        if attribute_separator is not None:
            gene_list = [s.split(attribute_separator)[0] for s in gene_list]

        df = self.getInteractionNamed()
        interactions_df = df.loc[df.Gene_A.isin(gene_list) &
                                 df.Gene_B.isin(gene_list)]
        return Graph(interactions_df)

    '''
        Get shortest path distance
    '''
    def getGeodesicDistance(self, start_genes, stop_genes, nx_Graph=None):
        pass

    def getAdjDict(self, return_names=True):
        pass

    def getEdgeArray(self):
        '''
        Return an edge array, a data structure that allows for
        a very fast retrieval of neighbors.
        :param return_names: Whether the array contains the node names (string) or ids (int)
        :return: edge array. a np array of arrays np.array([arr1, arr2, ..., arrN]) with arr1,
        containing the neighbors of the node with id 1 etc.
        '''
        adj_dict = self.getAdjDict(return_names=False)
        return np.array([np.array(adj_dict[i]) for i in range(self.N_nodes)])

    def getNOrderNeighbors(self, order=2, include_lower_order=True, gene_list=None):

        adj_dict = copy.deepcopy(self.getAdjDict())
        orig_dict = self.getAdjDict()

        if gene_list is not None:
            adj_dict = {k: v for k, v in adj_dict.items() if k in gene_list}

        for _ in range(order-1):
            adj_dict = getSecondOrderNeighbors(adj_dict, adj_dict0=orig_dict,
                                               incl_first_order=include_lower_order)
        return adj_dict

    def getDegreeDF(self, return_names=True, set_index=False):
        v, c = np.unique(self.interactions.values.flatten(), return_counts=True)
        if return_names:
            if set_index:
                return pd.DataFrame({'Gene': [self.int2gene[i] for i in v],
                                     'Count': c}, index=[self.int2gene[i] for i in v]).sort_values(by='Count',
                                                                                                   ascending=False,
                                                                                                   inplace=False)
            else:
                return pd.DataFrame({'Gene': [self.int2gene[i] for i in v],
                                     'Count': c}).sort_values(by='Count', ascending=False, inplace=False)
        else:
            if set_index:
                return pd.DataFrame({'Gene': v,
                                     'Count': c}, index=v).sort_values(by='Count', ascending=False, inplace=False)
            else:
                return pd.DataFrame({'Gene': v,
                                 'Count': c}).sort_values(by='Count', ascending=False, inplace=False)

    def removeNodes(self, nodes_tbr, inplace=False):
        nodes_tbr = [self.gene2int[s] for s in nodes_tbr if s in self.gene2int.keys()]
        nodes_tbr = set(nodes_tbr)

        if inplace:
            self.interactions = self.interactions.loc[~(self.interactions.Gene_A.isin(nodes_tbr) |
                                                        self.interactions.Gene_B.isin(nodes_tbr))]
            self.remove_nodes_from_type_dict(nodes_tbr)

        else:
            new_df = self.interactions.loc[~(self.interactions.Gene_A.isin(nodes_tbr) |
                                             self.interactions.Gene_B.isin(nodes_tbr))]
            new_df = new_df.applymap(lambda x: self.int2gene[x])

            return new_df

    def replaceNodesWithInteractions(self, nodes_tbr):
        '''
        Replaces a list of nodes, while connecting all neighbors to each other.
        To be used in pathfinding.
        :param nodes_tbr:
        :return:
        '''

        def get_all_nb_combos(neighbors):

            neighbors = [(nb1, nb2) for i, nb1 in enumerate(neighbors) for nb2 in neighbors[:i]]

            if len(neighbors) > 0:
                neighbors = np.array(neighbors)
                return pd.DataFrame(neighbors, columns=['Gene_A', 'Gene_B'])

            else:
                return None

        df_filtered = self.getInteractionNamed()
        adj_dict = self.getAdjDict()
        adj_dict = {k: np.array(v) for k, v in adj_dict.items()}
        nodes_tbr = set(nodes_tbr)
        new_interactions = []
        for node in nodes_tbr:

            new_interactions_ = get_all_nb_combos(adj_dict[node])
            adj_dict = {k: v[v != node] for k, v in adj_dict.items() if k != node}

            if new_interactions_ is not None:
                new_interactions.append(new_interactions_)

        df_filtered = df_filtered.loc[~(df_filtered.Gene_A.isin(nodes_tbr) |
                                      df_filtered.Gene_B.isin(nodes_tbr))]

        if len(new_interactions) > 0:
            new_interactions = pd.concat(new_interactions, axis=0)
            return pd.concat([df_filtered, new_interactions], axis=0)

        else:
            return df_filtered

    def filterDatasetGenes(self, omicsdatasets, remove_leaves=False, inplace=True):
        '''
        :param: omicsdatasets: datasets that are to be filtered
        :params: should the leaves of the network also be removed?
        :return: the filtered datasets whose genes are all on the network
        '''

        try:
            _ = len(omicsdatasets)

        except TypeError:  # convert to iterable
            omicsdatasets = [omicsdatasets]
            print('converted to iterable')

        if inplace:
            network_genes = set(self.node_names)
            nodes_in_datasets = set()

            for dataset in omicsdatasets:
                intersecting_genes = network_genes.intersection(dataset.genes(as_set=True))

                print('%s: %i genes found on the network.' % (dataset.type, len(intersecting_genes)))
                dataset.subsetGenes(list(intersecting_genes), inplace=True)

                network_genes = nodes_in_datasets.union(network_genes)
                nodes_in_datasets = nodes_in_datasets.union(dataset.genes(as_set=True))

            if remove_leaves:
                self.pruneNetwork(exception_list=nodes_in_datasets, inplace=True)

        else:
            network_genes = set(self.node_names)
            nodes_in_datasets = set()

            datasets_new = []

            for dataset in omicsdatasets:
                intersecting_genes = network_genes.intersection(dataset.genes(as_set=True))

                print('%s: %i genes found on the network.' % (dataset.type, len(intersecting_genes)))
                datasets_new += [dataset.subsetGenes(list(intersecting_genes), inplace=False)]

                network_genes = nodes_in_datasets.union(network_genes)
                nodes_in_datasets = nodes_in_datasets.union(dataset.genes(as_set=True))

            if remove_leaves:
                network = self.pruneNetwork(exception_list=nodes_in_datasets, inplace=False)

            return network, datasets_new

    def filterDataset(self, dataset, remove_leaves=False, inplace=False):

        keeps = list(set(self.node_names).intersection(dataset.genes(as_set=True)))

        if inplace:
            dataset.subsetGenes(keeps, inplace=True)

            if remove_leaves:
                self.pruneNetwork(exception_list=keeps, inplace=True)

        else:
            dataset = dataset.subsetGenes(keeps, inplace=False)
            net = self.deepcopy()
            if remove_leaves:
                net = net.pruneNetwork(exception_list=keeps, inplace=False)

            return net, dataset

    def pruneNetwork(self, exception_list=[], inplace=False):
        '''
        Iteratively prunes the network such that leaves of the network are removed
        '''
        if inplace:
            net = self
        else:
            net = self.deepcopy()

        degreedf = net.getDegreeDF()
        leaves = set(degreedf.Gene[degreedf.Count < 2])
        tbr = leaves.difference(set(exception_list))

        while len(tbr) > 0:

            net.removeNodes(tbr, inplace=True)
            degreedf = net.getDegreeDF()

            leaves = set(degreedf.Gene[degreedf.Count < 2])
            tbr = leaves.difference(exception_list)

        if not inplace:
            return net

    def getEdgeSet(self):
        df = self.getInteractionNamed()
        return set(zip(df['Gene_A'].values, df['Gene_B'].values))

    def getOverlap(self, other_net):
        set1 = self.getEdgeSet()
        set2 = other_net.getEdgeSet()

        min_len = np.minimum(len(set1), len(set2))

        return len(set1.intersection(set2))/min_len

    def keepLargestComponent(self, verbose=True, inplace=False):

        if self.isConnected:
            print('Graph is connected, returning a copy.')
            return self.deepcopy()

        else:
            components = self.getComponents(return_subgraphs=True)
            largest_subnet = max(components, key=len)

            if verbose:
                print('%i genes from smaller components have been removed.' % (self.N_nodes - largest_subnet.N_nodes))

            if inplace:
                self.setEqual(largest_subnet)
            else:
                return max(components, key=len)

    def getnxGraph(self, return_names=True):
        '''return a graph instance of the networkx module'''
        if return_names:
            df = self.getInteractionNamed()
        else:
            df = self.interactions

        return nx.from_pandas_edgelist(df, source='Gene_A', target='Gene_B')

    def getMinimmumSpanningTree(self, return_names=True):
        A = self.getnxGraph(return_names=return_names)
        T = nx.minimum_spanning_tree(A.to_undirected())

        E = T.edges()
        self_edges = set(self.edge_list(return_names=return_names))
        # assure that the edges are in the same order as the original Graph
        return [e if e in self_edges else tuple(reversed(e)) for e in E]

    def subsample(self, n=100, weighted=False):
        if weighted:
            v, c = np.unique(self.getInteractionNamed().values, return_counts=True)
            genes = np.random.choice(v, size=n, replace=False, p=c/np.sum(c))

        else:
            v = np.unique(self.getInteractionNamed())
            genes = np.random.choice(v, size=n, replace=False)

        subset_df = self.subsetNetwork(genes, inplace=False)

        return subset_df

    def getSimpleRepresentation(self, **kwargs):
        pass
    
    def get_degree_binning(self, bin_size=100, degree_to_nodes=None, return_names=True):
        '''
        code taken from Network-based in silico drug efficacy screening
        (https://github.com/emreg00/toolbox/blob/master/network_utilities.py)
        :param bin_size:
        :param degree_to_nodes: (optional) a precomputed dict with degrees as keys and node lists as value
        :return:
        '''
        if degree_to_nodes is None:
            degrees = self.getDegreeDF(return_names=return_names)
            unique_degrees = np.unique(degrees.Count.values)
            genes = degrees.index.values
            counts = degrees.Count.values
            degree_to_nodes = {i: list(genes[counts == i]) for i in unique_degrees}

        values = list(degree_to_nodes.keys())
        values.sort()
        bins = []
        i = 0
        while i < len(values):
            low = values[i]
            val = degree_to_nodes[values[i]]
            while len(val) < bin_size:
                i += 1
                if i == len(values):
                    break
                val.extend(degree_to_nodes[values[i]])
            if i == len(values):
                i -= 1
            high = values[i]
            i += 1
            # print i, low, high, len(val)
            if len(val) < bin_size:
                low_, high_, val_ = bins[-1]
                bins[-1] = (low_, high, val_ + val)
            else:
                bins.append((low, high, val))
        return bins

    def calculate_proximity_significance(self, nodes_from, nodes_to, shuffle_strategy='nodes_to',
                                        n_random=1000, min_bin_size=100, seed=452456, measure='d_c'):
        """
        Calculate proximity from nodes_from to nodes_to
        If degree binning or random nodes are not given, they are generated
        lengths: precalculated shortest path length dictionary
        """

        np.random.seed(seed)
        nodes_network = set(self.node_names)

        nodes_from = set(nodes_from) & nodes_network
        nodes_to = set(nodes_to) & nodes_network
        gene2int = self.gene2int

        nodes_from = [gene2int[g] for g in nodes_from]
        nodes_to = [gene2int[g] for g in nodes_to]

        if len(nodes_from) == 0 or len(nodes_to) == 0:
            return None  # At least one of the node group not in network

        nx_Graph = self.getnxGraph(return_names=False)

        d = self.calculate_dist_from_group(nodes_from, nodes_to, measure=measure, nx_Graph=nx_Graph)

        if shuffle_strategy.lower() == 'nodes_to':
            node_to_equivalents = self.get_degree_equivalents(seeds=nodes_to, bin_size=min_bin_size,
                                                              return_names=False)
            nodes_to = select_random_sets(node_to_equivalents, nodes_to, n_random)
            random_dists = [self.calculate_dist_from_group(nodes_from, nodes_to[i, :], nx_Graph=nx_Graph) for i in range(n_random)]

        elif shuffle_strategy.lower() == 'nodes_from':
            node_to_equivalents = self.get_degree_equivalents(seeds=nodes_from, bin_size=min_bin_size,
                                                              return_names=False)
            nodes_from = select_random_sets(node_to_equivalents, nodes_from, n_random)
            random_dists = [self.calculate_dist_from_group(nodes_from[i, :], nodes_to, nx_Graph=nx_Graph) for i in range(n_random)]

        elif shuffle_strategy.lower() == 'both':
            node_to_equivalents = self.get_degree_equivalents(seeds=np.union1d(nodes_to, nodes_from),
                                                              bin_size=min_bin_size,
                                                              return_names=False)
            nodes_to = select_random_sets(node_to_equivalents, nodes_to, n_random)
            nodes_from = select_random_sets(node_to_equivalents, nodes_from, n_random)
            random_dists = [self.calculate_dist_from_group(nodes_from[i, :], nodes_to[i, :], nx_Graph=nx_Graph) for i in range(n_random)]

        else:
            raise IOError('shuffle_strategy not understood.')

        pval = float(sum(random_dists <= d)) / len(random_dists)  # needs high number of n_random
        m, s = np.mean(random_dists), np.std(random_dists)
        if s == 0:
            z = 0.0
        else:
            z = (d - m) / s

        return d, z, (m, s), pval

    def get_degree_equivalents(self, seeds, bin_size=100, return_names=True):

        if return_names:
            seeds = np.intersect1d(seeds, self.node_names)
        else:
            seeds = np.intersect1d(seeds, self.nodes)

        if len(seeds) == 0:
            raise IOError('The seeds do not match the names of the graph nodes.')

        degrees = self.getDegreeDF(return_names=return_names, set_index=True)
        unique_degrees = np.unique(degrees.Count.values)
        genes = degrees.index.values
        counts = degrees.Count.values
        degree_to_nodes = {i: list(genes[counts == i]) for i in unique_degrees}

        bins = self.get_degree_binning(bin_size=bin_size, degree_to_nodes=degree_to_nodes,
                                       return_names=return_names)
        seed_to_nodes = {}

        for seed in seeds:
            d = counts[genes == seed]

            for l, h, nodes in bins:
                if (l <= d) and (h >= d):
                    mod_nodes = list(nodes)
                    mod_nodes.remove(seed)
                    seed_to_nodes[seed] = mod_nodes
                    break

        return seed_to_nodes

    def calculate_dist_from_group(self, nodes_from, nodes_to, measure='d_c', nx_Graph=None):
        dist_mat = self.getGeodesicDistance(nodes_from, nodes_to, nx_Graph=nx_Graph)

        if measure == 'd_c':
            min_dists = np.min(dist_mat.values, axis=1)
            return np.mean(min_dists)

        elif measure == 'd_s':
            mean_dists = np.mean(dist_mat.values, axis=1)
            return np.mean(mean_dists)

    def visualize(self, return_large=False, gene_list=None, edge_df=None, show_labels=False,
                  node_colors=None, cmap='spectral', title=None,
                  color_scheme_nodes=('lightskyblue', 'tab:orange'),
                  color_scheme_edges=('gray', 'tab:green'), labels_dict=None,
                  filename=None, save_path=None):

        """ Visualize the graph
         gene_list = MUST be a list of lists
         labels_dict: a dictionary of dictionaries, containing the labels, fontsizes etc for each group of labels.

         example: {'group1': {'labels': ['G1', 'G2'],
                         font_size:12,
                         font_color:'k',
                         font_family:'sans-serif',
                         font_weight:'normal',
                         alpha:None,
                         bbox:None,
                         horizontalalignment:'center',
                         verticalalignment:'center'}}

        note that the name of the keys is not used.
         """

        if gene_list is not None:
            assert len(gene_list) == len(color_scheme_nodes)-1, \
                "ERROR number of gene lists provided must match the color scheme for nodes"

        if (not return_large) and (len(self.nodes) > 500):
            raise IOError('The graph contains more than 500 nodes, if you want to plot this specify return_large=True.')

        G = self.getnxGraph()
        if (gene_list is None) and (node_colors is None):
            node_colors = color_scheme_nodes[0]
        elif node_colors is None:
            additional_gl = set.intersection(*[set(i) for i in gene_list])
            if additional_gl:
                gene_list = [set(gl)-additional_gl for gl in gene_list]
                gene_list.append(additional_gl)
                color_scheme_nodes += ("tab:purple",)
            node_colors = []
            for i, gl in enumerate(gene_list):
                node_colors.append([color_scheme_nodes[i+1] if node in gl else "" for node in G.nodes])
            node_colors = list(map(''.join, zip(*node_colors)))
            node_colors = [i if i else color_scheme_nodes[0] for i in node_colors]
            # node_colors = [color_scheme_nodes[1] if node in gene_list else color_scheme_nodes[0] for node in G.nodes]

        # assert len(G.nodes) == len(node_colors), "ERROR number of node colors does not match size of graph"

        if all(isinstance(c, (int, float)) for c in node_colors):  # perform rescaling in case of floats for cmap
            node_colors = np.array(node_colors)
            node_colors = (node_colors - np.min(node_colors))/(np.max(node_colors) - np.min(node_colors))

        if edge_df is not None:
            edges = list(G.edges())
            edge_list = [tuple(pair) for pair in edge_df.values]

            edge_color = [color_scheme_edges[1] if edge in edge_list else color_scheme_edges[0] for edge in edges]
            edge_thickness = [2 if edge in edge_list else 1 for edge in edges]

        else:
            edge_color = color_scheme_edges[0]
            edge_thickness = 1.

        plt.figure()
        # TODO: make this prettier
        if title is not None:
            plt.title(title)

        if labels_dict is None:
            nx.draw(G, with_labels=show_labels,
                    node_size=2e2, node_color=node_colors, edge_color=edge_color, width=edge_thickness, cmap=cmap)

        else:
            pos = nx.drawing.spring_layout(G)  # default to spring layout

            nx.draw(G, pos=pos, with_labels=False,
                    node_size=2e2, node_color=node_colors,
                    edge_color=edge_color, width=edge_thickness, cmap=cmap)

            for label_kwds in labels_dict.values():
                nx.draw_networkx_labels(G, pos, **label_kwds)

        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = [20, 15]

        if filename:
            plt.savefig(save_path + filename + '.png')
            plt.close()
        else:
            plt.show()

    def plotRepresentation(self, gene_list=None, node_colors =None, cmap='spectral', precomputed_representations=None,
                           change_size=True, **kwargs):
        '''
        Function to plot the network without the edges, resulting in a much faster visualization, making it possible to visualize larger graphs
        :param gene_list: list of genes to be plotted
        :param node_colors: color of the nodes can be strings with color or floats
        :param node_size:
        :param: cmap:
        :param: precomputed_representations:
        :return:
        '''
        size = None

        if node_colors is None:
            node_colors = 'lightskyblue'

        if all(isinstance(c, (int, float)) for c in node_colors):  # perform rescaling in case of floats for cmap
            node_colors = np.array(node_colors)
            node_colors = (node_colors - np.min(node_colors))/(np.max(node_colors) - np.min(node_colors))

            if change_size:
                size = 50 * node_colors

        if precomputed_representations is not None:
            coords = precomputed_representations
        else:
            coords = self.getSimpleRepresentation(dim=2)

        if gene_list is not None:
            coords = coords.loc[gene_list]

        fig, ax = plt.subplots()
        ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=node_colors, s=size, **kwargs)
        ax.set_visible('off')
        plt.show()

    def degreePreservingPermutation(self, N_swaps=1000, random_state=42):
        np.random.seed(42)
        adj_dict = self.getAdjDict(return_names=False)
        adj_dict = {k: np.array(v) for k, v in adj_dict.items()}

        weights = self.getDegreeDF(return_names=False).set_index('Gene')['Count']
        weights = 1. * weights.loc[np.arange(self.N_nodes)].values/np.sum(weights.values)
        n_swaps = 0
        visited_pairs = []

        while n_swaps < N_swaps:
            gene1, gene2 = np.random.choice(self.N_nodes, 2, replace=False, p=weights)

            if (gene1, gene2) not in visited_pairs:
                nb_gene1, nb_gene2 = copy.deepcopy(adj_dict[gene1]), copy.deepcopy(adj_dict[gene2])

                connected = (gene1 in nb_gene2) | (gene2 in nb_gene1)  #TODO: check if this works for Directed Graphs

                if connected:
                    nb_gene1 = np.append(nb_gene1, gene1)
                    nb_gene2 = np.append(nb_gene2, gene2)

                overlap = np.intersect1d(nb_gene1, nb_gene2)

                if (len(nb_gene2) > len(overlap)) & (len(nb_gene1) > len(overlap)):

                    if len(nb_gene1) < len(nb_gene2):  # make sure nb1 has the most neighbors
                        t_ = nb_gene2
                        nb_gene2 = nb_gene1
                        nb_gene1 = t_

                        t_ = gene2
                        gene2 = gene1
                        gene1 = t_

                    diff = np.setdiff1d(nb_gene1, nb_gene2)  # append gene1 in case gene1 -- gene2

                    n_swapped_genes = len(nb_gene2) - len(overlap)

                    random_ids = np.random.choice(len(diff), n_swapped_genes, replace=False)
                    one_to_two = diff[random_ids]
                    two_to_one = np.setdiff1d(nb_gene2, nb_gene1)

                    arr2 = np.union1d(overlap, one_to_two)
                    arr1 = np.union1d(nb_gene2, np.delete(diff, random_ids))

                    if connected:
                        arr2 = arr2[arr2 != gene2]
                        arr1 = arr1[arr1 != gene1]

                    adj_dict[gene2] = copy.deepcopy(arr2)
                    adj_dict[gene1] = copy.deepcopy(arr1)

                    for a in one_to_two:
                        adj_dict[a][adj_dict[a] == gene1] = gene2

                    for a in two_to_one:
                        adj_dict[a][adj_dict[a] == gene2] = gene1

                    visited_pairs.append((gene1, gene2))
                    visited_pairs.append((gene2, gene1))
                    n_swaps += 1

        df = adj_dict_to_df(adj_dict)
        df = df.applymap(lambda x: self.int2gene[x])

        return df
    
    def getTrainTestPairs_MStree(self, train_ratio=0.7, train_validation_ratio=0.7,
                                 excluded_sets=None, neg_pos_ratio=5, check_training_set=True, random_state=42):
        pass

    def getTrainTestData(self, train_ratio=0.7, neg_pos_ratio=5, train_validation_ratio=None, excluded_sets=None,
                         return_summary=True, random_state=42):
        '''
        :param: train_ratio: The fraction of samples used for training
        :param: neg_pos_ratio: The ratio of negative examples to positive examples
        :param: excluded_negatives: should be a set of tuples with negatives interactions to exclude
        :param: method: The sampling method used for generating the pairs:
                - ms_tree: uses a minimum spanning tree to find at least one positive pair for each node
                - balanced: draws approximately (neg_pos_ratio * n_positives) negatives for each gene
        :return: positive and negative pairs for both train and test set (4 lists in total)
        '''
        pos_train, neg_train, pos_val, neg_val, \
        pos_test, neg_test, summary_df = self.getTrainTestPairs_MStree(train_ratio=train_ratio,
                                                                       train_validation_ratio=train_validation_ratio,
                                                                       excluded_sets=excluded_sets,
                                                                       neg_pos_ratio=neg_pos_ratio,
                                                                       check_training_set=True,
                                                                       random_state=random_state)

        assert len(pos_train) == len(set(pos_train)), "getTrainTestPairs_MStree: Duplicate pos train"
        assert len(pos_test) == len(set(pos_test)), "getTrainTestPairs_MStree: Duplicate pos test"

        assert len(neg_train) == len(set(neg_train)), "getTrainTestPairs_MStree: Duplicate neg train"
        assert len(neg_test) == len(set(neg_test)), "getTrainTestPairs_MStree: Duplicate neg test"

        assert not set(pos_train) & set(pos_test), "getTrainTestPairs_MStree: Overlap pos train - test"
        assert not set(neg_train) & set(neg_test), "getTrainTestPairs_MStree: Overlap neg train - test"

        if len(pos_val) > 1:
            assert not set(pos_val) & set(pos_test), "getTrainTestPairs_MStree: Overlap pos val - test"
            assert not set(neg_val) & set(neg_test), "getTrainTestPairs_MStree: Overlap neg val - test"

        X_train = np.array(pos_train + neg_train)
        X_val = np.array(pos_val + neg_val)
        X_test = np.array(pos_test + neg_test)
    
        Y_train = np.array([1 for _ in range(len(pos_train))] + [0 for _ in range(len(neg_train))])
        Y_val = np.array([1 for _ in range(len(pos_val))] + [0 for _ in range(len(neg_val))])
        Y_test = np.array([1 for _ in range(len(pos_test))] + [0 for _ in range(len(neg_test))])
        
        if return_summary:
            if train_validation_ratio is None:
                return X_train, X_test, Y_train, Y_test, summary_df
            else:
                return X_train, X_val, X_test, Y_train, Y_val, Y_test, summary_df
        else:
            if train_validation_ratio is None:
                return X_train, X_test, Y_train, Y_test
            else:
                return X_train, X_val, X_test, Y_train, Y_val, Y_test


class DirectedInteractionNetwork(Graph):
    def __init__(self, interaction_df, colnames=None, verbose=True, keeplargestcomponent=False,
                 allow_self_connected=False, node_types=None, gene2int=None):
        super().__init__(interaction_df, colnames, verbose=verbose, keeplargestcomponent=keeplargestcomponent,
                         allow_self_connected=allow_self_connected, node_types=node_types, gene2int=gene2int)

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

    def getUndirectedInteractionNetwork(self):
        return UndirectedInteractionNetwork(self.getInteractionNamed(),
                                            node_types=self.node_type_names)

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
        #TODO: update this function for validation split
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



class NBDInet(UndirectedInteractionNetwork):

    def __init__(self, interaction_df, omicsdatasets, colnames_interactiondf=None):

        temp = Graph(interaction_df, colnames=colnames_interactiondf, verbose=False)
        gene_names_net = temp.nodes

        ids_dict = {node: node + ' NET' for node in temp.nodes}
        temp.changeIDs(ids_dict=ids_dict)

        datatypes, samples = ['NET'], []
        dfs = [temp.interactions]

        for dataset in omicsdatasets:
            genes_in_network = set(gene_names_net).intersection(dataset.genes(as_set=True))
            datadf = dataset.get_all_pairs(include_type=True, colnames=('Gene_A', 'Gene_B'))

            dfs += [datadf]  # Add the Sample to status node interactions
            dfs += [pd.DataFrame({'Gene_A': [gene + ' NET' for gene in genes_in_network],
                                  'Gene_B': [gene + ' ' + dataset.type for gene in genes_in_network]})]
            datatypes.append(dataset.type)

            if (len(set(samples).intersection(dataset.samples(as_set=True))) < 1) & (len(samples) > 0):
                warnings.warn('No overlap with samples from %s dataset, be sure to use the correct identifiers' % dataset.type, UserWarning)

            samples += list(set([s.split(' ')[0] for s in datadf.Gene_A]))

        # Connect the status node to the node in the network

        super().__init__(pd.concat(dfs, axis=0), verbose=True)
        self.datatypes = datatypes
        self.samples = list(set(samples))
        self.diffused = None

        print('Integrated %i datatypes and %i samples.' % (len(self.datatypes), len(self.samples)))

    def extractSubmatrix(self, Suffix, diffused_mat=False):
        '''
        :param Suffix: A suffix  tuple(' EXP', ' MUT', ' NET', ...) which has to be selected  from the matrix
        :return: The submatrix containing all nodes suffix[0] x all nodes suffix[1]
        '''
        Suffix_A, Suffix_B = set([Suffix[0]]), set([Suffix[1]])

        if diffused_mat:
            if self.diffused is not None:
                mask_A = np.array([gene.split(' ')[-1] in Suffix_A for gene in self.nodes])  #TODO: get rid of these magic numbers
                mask_B = np.array([gene.split(' ')[-1] in Suffix_B for gene in self.nodes])

                return pd.DataFrame(self.diffused[mask_A, :][:, mask_B],
                                    index=np.array(self.nodes)[mask_A],
                                    columns=np.array(self.nodes)[mask_B])
            else:
                raise ValueError('The matrix is not yet diffused')

        else:  # return a subset of the adjacency matrix,
            mask = np.array(self.interactions.Gene_A.apply(lambda x: x[-4:]).isin(set(Suffix_A))) | \
                   np.array(self.interactions.Gene_B.apply(lambda x: x[-4:]).isin(set(Suffix_B)))

            return UndirectedInteractionNetwork(self.interactions.loc[mask]).getAdjMatrix(sort='last', as_df=True)

    def diffuse(self, kernel='LEX', alpha=0.01, as_df=False):
        self.diffused, self.nodes = super().diffuse(kernel=kernel, alpha=alpha, as_df=False)

    def getTopFeatures(self, data_types=None, topn=10, return_values=False, sample_groups=None, dist_measure='mean'):
        '''
        :param data_type: indicates the data types that need to be considered (EXP, MUT, DEL, AMP)
        :param topn: the number of top features considered for each data type
        :return: a list of len(data_types) x topn elements
        '''

        if data_types is None:
            data_types = self.datatypes
        else:
            data_types = set(data_types).intersection(set(self.datatypes))
            if len(data_types) < 1:
                warnings.warn('Types do not overlap with the datatypes in the NBDI network.', UserWarning)

        topfeatures = {}
        values = []

        for data_type in data_types:
            Dist = self.extractSubmatrix(('PAT', data_type), diffused_mat=True)
            features_ = Dist.columns.values

            if (dist_measure.lower() == 'mean_pos_neg') & (sample_groups is not None):
                # for now we assume two classes only
                sample_groups = sample_groups.loc[[s.split(' ')[0] for s in list(Dist.index)]]
                classes = pd.unique(sample_groups)

                if len(classes) != 2:
                    raise NotImplementedError('Feature selection can only happen for 2 classes.')

                mean_dist1 = Dist.loc[sample_groups.values == classes[0]].mean(axis=0)
                mean_dist2 = Dist.loc[sample_groups.values == classes[1]].mean(axis=0)

                dist = 2*(mean_dist1 - mean_dist2).abs()/(mean_dist1 + mean_dist2)
                mean_ids = dist.values.argsort()[-topn:][::-1]

            else:
                dist = Dist.mean(axis=0)
                mean_ids = dist.values.argsort()[-topn:][::-1]

            topfeatures[data_type] = [s[:s.rfind(' ')] for s in features_[mean_ids]]
            values += [dist]

        values_df = pd.concat(values, axis=0)
        if return_values:
            return topfeatures, values_df
        else:
            return topfeatures


## Helper functions


def adj_dict_to_df(adj_dict):

    data = np.array([(k, v) for k, vals in adj_dict.items() for v in vals])
    return pd.DataFrame(data, columns=['Gene_A', 'Gene_B'])


def getSecondOrderNeighbors(adj_dict, adj_dict0=None, incl_first_order=True):
    # slwo
    if adj_dict0 is None:
        adj_dict0 = adj_dict

    if incl_first_order:
        return {k: set([l for v_i in list(v) + [k] for l in adj_dict0[v_i]]) for k, v in adj_dict.items()}
    else:
        return {k: set([l for v_i in v for l in adj_dict0[v_i]]) for k, v in adj_dict.items()}


def extractIDsfromBioGRID(path_df, one2one=True):
    '''
    Creates a map (dictionary) between entrez and gene symbols (including aliases) from a BioGRID interaction file.
    :param path_df: path or dataframe containing the interaction data from BioGRID
    :param one2one: whether the resulting map should map str to str or str to tuple
    :return:
    '''

    if isinstance(path_df, str):
        biogrid_df = pd.read_csv(path_df, sep='\t', header=0)

    elif isinstance(path_df, pd.DataFrame):
        biogrid_df = path_df

    else:
        raise IOError('Input type not understood.')

    try:
        biogrid_df = biogrid_df[['Official Symbol Interactor A', 'Official Symbol Interactor B', 'Synonyms Interactor A',
                                 'Synonyms Interactor B', 'Entrez Gene Interactor A', 'Entrez Gene Interactor B']]

    except KeyError:
        raise IOError('The dataframe does not contain the BioGRID column names.')

    # make the entrez to symbol map:
    map_df = pd.DataFrame({'Entrez': biogrid_df['Entrez Gene Interactor A'].append(biogrid_df['Entrez Gene Interactor B']),
                        'Gene Symbol': biogrid_df['Official Symbol Interactor A'].append(biogrid_df['Official Symbol Interactor B'])})

    map_df = map_df.drop_duplicates()
    map_dict = dict(zip(map_df.Entrez, map_df['Gene Symbol']))

    # to make tuple maps
    # symbols = biogrid_df['Official Symbol Interactor A'].append(biogrid_df['Official Symbol Interactor B'])
    # aliases = biogrid_df['Synonyms Interactor A'].append(biogrid_df['Synonyms Interactor B'])

    # tuples = (symbols.astype(str) + '|' + aliases.astype(str)).apply(lambda x: tuple(x.replace('|-', '').split('|')))

    return map_dict

# monkey patching at its ugliest :)


def from_df(self, df, weighted=False, directed=False, weights_col=None):
    self.G = nx.DiGraph()
    src_col = df.columns[0]
    dst_col = df.columns[1]
    if directed:
        def read_weighted(src, dst, w):
            self.G.add_edge(src, dst)
            self.G[src][dst]['weight'] = w

    else:
        def read_weighted(src, dst, w):
            self.G.add_edge(src, dst)
            self.G.add_edge(dst, src)
            self.G[src][dst]['weight'] = w
            self.G[dst][src]['weight'] = w
    
    if weights_col is None:
        weights = [1.0 for row in range(df.shape[0])]
    else:
        try:
            weights = df[weights_col].values.astype(float)
        except:
            raise IOError('The weight column is not known.')

    for src, dst, w in zip(df[src_col].values, df[dst_col].values, weights):
        read_weighted(src, dst, w)

    self.encode_node()


def select_random_sets(equivalency_dict, node_group, nsets, seed=42):
    '''
    draws samples from the equivalency dict such that we end up with nsets sets.
    Each set contains unique elements sampled from the equivalency dict.
    :param equivalency_dict: a dict with elements from nsets as key and valid replacements as values (list)
    :param node_group:
    :param nsets: the number of sets available
    :return: an (len(node_group, nsets) np.array
    '''

    np.random.seed(seed)

    group_size = len(node_group)
    bin_sizes = [len(v) for v in equivalency_dict.values()]
    min_bin_size = min(bin_sizes)
    most_frequent_bin, max_bin_freq = np.unique(bin_sizes, return_counts=True)
    max_bin_freq = max_bin_freq[np.argmax(most_frequent_bin)]

    #p_unique = np.prod(np.arange(min_bin_size, min_bin_size - max_bin_freq, -1)/min_bin_size)
    #p_unique = (min_bin_size - group_size + 1)/(min_bin_size - 1)
    #ndraws = np.int(nsets / p_unique * 1.1)

    rand_sets = np.array([np.random.choice(equivalency_dict[g], nsets, replace=True)
                          for g in node_group])

    rand_sets = np.transpose(rand_sets)
    mask = np.array([len(np.unique(gs)) == group_size for gs in rand_sets])
    rand_sets = rand_sets[mask]

    while rand_sets.shape[0] < nsets:
        still_needed = 2*(nsets - rand_sets.shape[0])
        rand_sets_ = np.array([np.random.choice(equivalency_dict[g], still_needed, replace=True)
                              for g in node_group])

        rand_sets_ = np.transpose(rand_sets_)
        mask = np.array([len(np.unique(gs)) == group_size for gs in rand_sets])
        rand_sets = rand_sets[mask]
        rand_sets = np.vstack((rand_sets, rand_sets_))

    return rand_sets[:nsets, :]


def check_node_dict(net_node_names, node_dict, type_dict=""):
    """
    expects a dict where keys = nodes in the network, values are the corresponding types
    """

    nodes = list(node_dict.keys())

    nodes_missing_in_dict = set(net_node_names) - set(nodes)

    if len(nodes_missing_in_dict) > 0:
        print("The following genes have no annotation:")
        print(nodes_missing_in_dict)

        raise IOError("There are genes that are missing in the %s dict." %type_dict)

    nodes_missing_in_network = set(nodes) - set(net_node_names)

    if len(nodes_missing_in_network) > 0:
        # print("The following genes are missing in the network and will be removed:")
        # print(nodes_missing_in_network)
        return {k: v for k, v in node_dict.items() if k not in nodes_missing_in_network}
    else:
        return node_dict


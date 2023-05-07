import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import copy
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

class Tree:
    def __init__(self, expression_data, linkage='average'):
        self.similarity_matrix = expression_data.df.corr(method='pearson').abs()
        self.model = AgglomerativeClustering(n_clusters=2, linkage=linkage, compute_full_tree=True)
        self.model.fit_predict(self.similarity_matrix)

        self.hierarchy = self.generate_hierarchy()
        self.nodes = self.hierarchy.index
        self.hierarchy['objects'] = [self.get_children_objects(node) for node in self.nodes]

    def generate_hierarchy(self):
        ii = itertools.count(self.similarity_matrix.shape[1])
        hierarchy = [{'node_id': next(ii), 'left': x[0], 'right': x[1]} for x in self.model.children_]
        hierarchy = pd.DataFrame(hierarchy)
        hierarchy = hierarchy.set_index('node_id')
        return hierarchy

    def get_children(self, node):
        children = self.hierarchy.loc[node, ['left', 'right']]
        return children

    def get_children_objects(self, node, objects=set()):
        objects_ = copy.copy(objects)
        for child in self.get_children(node):
            if child in self.nodes:
                objects_ = self.get_children_objects(child, objects_)
            else:
                objects_.add(child)
        return objects_

    def get_sigma(self, node, reference_tree, sigma=0):
        own_sigma = copy.copy(sigma)
        for child in self.get_children(node):
            if child in self.nodes:
                own_sigma += self.get_sigma(child, reference_tree, sigma)
        # Check if ref tree has a node containing the same objects
        objects = self.hierarchy.loc[node, 'objects']
        if objects in list(reference_tree.hierarchy.loc[:, 'objects']):
            own_sigma += 1/len(objects)
        return own_sigma

    def similarity_score(self, reference_tree):
        """
        calculates a score that indicates the difference between this tree and a reference tree, where 0 means
        completely different and 1 means exactly the same
        :param reference_tree: another Tree object
        :return: a score between 0 and 1
        """
        root = self.nodes[-1]
        children = self.get_children(root)
        BScore = self.get_sigma(children['left'], reference_tree) + \
                 self.get_sigma(children['right'], reference_tree)

        root = reference_tree.nodes[-1]
        children = reference_tree.get_children(root)
        refscore = reference_tree.get_sigma(children['left'], reference_tree) + \
                   reference_tree.get_sigma(children['right'], reference_tree)

        normalized_sim_score = BScore/refscore
        return normalized_sim_score

    def plot_dendogram(self, **kwargs):
        # Children of hierarchical clustering
        children = self.model.children_

        # Distances between each pair of children
        # Since we don't have this information, we can use a uniform one for plotting
        distance = np.arange(children.shape[0])

        # The number of observations contained in each cluster level
        no_of_observations = np.arange(2, children.shape[0] + 2)

        # Create linkage matrix and then plot the dendrogram
        linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

        # Plot the corresponding dendrogram
        fig, ax = plt.subplots()
        dendrogram(linkage_matrix, **kwargs)
        plt.show()

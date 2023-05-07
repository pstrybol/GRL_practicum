import pandas as pd
import numpy as np
import warnings
from statsmodels.stats.multitest import multipletests
from scipy.stats import hypergeom


class GeneSet:

    def __init__(self, dictionary):
        pathdict = {k: np.array([v for v in l if isinstance(v, str) and (v != '')]) for k, l in
                    dictionary.items()}
        # check that all pathways contain unique genes
        pathdict = {k: np.unique(v) for k, v in pathdict.items()}

        # remove empty pathways from the list
        self.pathdict = {k: v for k, v in pathdict.items() if len(v) > 0}

    @classmethod
    def from_df(cls, gmt_df, axis=1):
        # initialize from a gmt dataframe
        # axis denote the pathway axes, axis=1 means columns denote pathways
        try:
            rows, cols = gmt_df.shape
        except ValueError:
            raise ValueError('Please provide a 2-D dataframe as the input.')

        if axis == 0:
            gmt_df = gmt_df.transpose()
        elif axis != 1:
            raise IOError('Axis can only take values in {0, 1}.')

        pathways = gmt_df.columns.values

        if len(pathways) != len(np.unique(pathways)):
            raise IOError('Duplicate pathways names in the Input.')

        return cls(gmt_df.to_dict(orient='list'))

    @classmethod
    def from_file(cls, file):
        gmt_d = {}
        try:
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    temp = line.strip('\n').split('\t')
                    gmt_d[temp[0]] = set(gene for gene in temp[2:])
        except FileNotFoundError:
            print(f"Error file {file} not found")

        return cls(gmt_d)

    def __getitem__(self, id):
        try:
            out = self.pathdict[id]
        except KeyError:
            raise KeyError('The pathway is not present in this geneset.')

        return out

    def pathways(self, return_type='np'):
        pathways = list(self.pathdict.keys())
        if 'set' in return_type.lower():
            pathways = set(pathways)
        elif 'np' in return_type.lower():
            pathways = np.array(pathways)

        return pathways

    def genes(self, return_type='np'):
        pathways = [gene for path in self.pathdict.values() for gene in path]
        if 'set' in return_type.lower():
            pathways = set(pathways)
        elif 'np' in return_type.lower():
            pathways = np.unique(np.array(pathways))
        else:
            pathways = list(set(pathways))

        return pathways

    def subsetPathways(self, pathlist):
        return GeneSet({path: self.pathdict[path] for path in pathlist if path in self.pathways(return_type='set')})

    def getUnannotatedGenes(self, gene_list):
        # find genes that do not occur in any pathway
        gene_list = checkInputGenelist(gene_list)

        gene_list = np.array(gene_list)
        diff_genes = np.setdiff1d(gene_list, self.genes())

        print('There are %i genes that do not have an annotation.' % len(diff_genes))

        return diff_genes

    def getAnnotatedGenes(self, gene_list):
        gene_list = checkInputGenelist(gene_list)

        gene_list = np.array(gene_list)
        annotated_genes = np.intersect1d(self.genes(), gene_list)

        print('There are %i annotated genes' %len(annotated_genes))

        return annotated_genes

    def getAnnotationOnlyGenes(self, gene_list):
        gene_list = checkInputGenelist(gene_list)

        gene_list = np.array(gene_list)
        diff_genes = np.setdiff1d(self.genes(), gene_list)

        print('There are %i genes that have an annotation, but do not occur in the provided list.' % len(diff_genes))

        return diff_genes

    def getUniqueGenesList(self, gene_list=None):
        genes = self.genes()

        if gene_list is None:
            return gene_list
        else:
            gene_list = checkInputGenelist(gene_list)
            return np.union1d(genes, gene_list)

    def removeGenes(self, gene_list):
        # removes all genes in gene_list from the annotations
        gene_list = checkInputGenelist(gene_list)
        gene_list = set(gene_list)

        self.pathdict ={path: [gene for gene in l if gene not in gene_list] for path, l in self.pathdict.items()}

    def mapPathways(self, map, inplace=True):
        if inplace:
            self.pathdict = {map[path]:l for path, l in self.pathdict.items()}
        else:
            return GeneSet({map[path]: l for path, l in self.pathdict.items()})

    def mapAllIds(self, map, inplace=False):
        if inplace:
            self.pathdict = {map[path]: [map[gene] for gene in l] for path, l in self.pathdict.items()}
        else:
            return GeneSet({map[path]: [map[gene] for gene in l] for path, l in self.pathdict.items()})

    def compareAnnotations(self, geneset):
        # compare the pathways between genesets
        common_pathways = np.intersect1d(self.pathways(), geneset.pathways())

        output_dict = {'Pathways':common_pathways}
        len_self, len_other, overlap, rel_overlap = [], [], [], []

        for path in common_pathways:
            len_self += [len(self.pathdict[path])]
            len_other += [len(geneset.pathdict[path])]
            intersection = np.intersect1d(self.pathdict[path], geneset.pathdict[path])

            overlap += [len(intersection)]
            rel_overlap += [len(intersection)/np.minimum(len(self.pathdict[path]), len(geneset.pathdict[path]))]

        output_dict['Size geneset'] = len_self
        output_dict['Size provided geneset'] = len_other
        output_dict['Overlap'] = overlap
        output_dict['Relative Overlap'] = rel_overlap

        return pd.DataFrame(output_dict)

    def getOverlapMatrix(self):

        pathways = self.pathways(return_type='list')
        unvisited_pathways = [path for path in pathways]

        overlap_df = pd.DataFrame(np.zeros((len(pathways), len(pathways))),
                                  columns=pathways, index=pathways)

        for pathways in pathways:
            unvisited_pathways.remove(pathways)

            for other_pathway in unvisited_pathways:
                overlap = np.intersect1d(self.pathdict[pathways], self.pathdict[other_pathway])
                overlap_df.loc[pathways, other_pathway] = overlap
                overlap_df.loc[other_pathway, pathways] = overlap

        return overlap_df

    def performHypergeomTesting(self, gene_list, M, fdr_thresh=0.05, corr_method='fdr_bh', verbose=False):
        """
        Performs GSEA on a genelist
        :param gene_list: Genelist containing genes of interest. Note that these identifiers need to be the same as
        the ones used in self.pathdict
        :param M: Number of genes in the universe
        :param fdr_thresh: Multiple testing correction threshold
        :param corr_method: Correction method
        :return: dataframe
        """

        gene_list = checkInputGenelist(gene_list)

        N = len(gene_list)
        pval_list = []
        hm_list = []
        if verbose:
            print('Length of the gene list after cleaning: %i' % len(gene_list))
        results_df = {'Pathway': [], 'Pval': []}
        for path, members in self.pathdict.items():
            # print(path)

            n = len(members)
            overlap = np.intersect1d(gene_list, members)
            X = len(overlap)
            if verbose:
                print('Overlap: %i' % len(overlap))
            pval = hypergeom.sf(X - 1, M, n, N)
            pval_list.append(pval)
            hm_list.append(path)

            results_df['Pathway'] += [path]
            results_df['Pval'] += [pval]

        if corr_method is not None:
            reject, results_df['FDR'], _, _ = multipletests(results_df['Pval'], fdr_thresh, method=corr_method)

        return pd.DataFrame(results_df).sort_values(by='Pval')

    def getDF(self):
        all_pairs = np.array([(path, gene) for path, l in self.pathdict.items() for gene in l])
        return pd.DataFrame(all_pairs, columns=['Pathway', 'Gene'])

    def sampleNegatives(self, ratio=1):
        neg_pairs = []
        for path_ways in self.pathways():
            diff = np.setdiff1d(self.genes(), self[path_ways])

            n_samples = np.minimum(len(diff), np.int(ratio*len(path_ways)))

            pairs = np.vstack((np.random.choice(diff, n_samples, replace=False),
                               np.array([path_ways for _ in range(n_samples)])))

            neg_pairs += [np.transpose(pairs)]

        return np.vstack(tuple(neg_pairs))

    def getTrainingSet(self, neg_ratio=1):
       pos = self.getDF().values
       neg = self.sampleNegatives(ratio=neg_ratio)

       Y = np.hstack((np.ones(len(pos)), np.zeros(len(neg))))

       X = np.vstack((pos, neg))

       return X, Y

    def getTrainTestSet(self, train_ratio=0.7, negpos_ratio=1):
        X_train, X_test, Y_train, Y_test = [], [], [], []
        for pathway in self.pathways():
            pos = np.array([(pathway, gene) for gene in self[pathway]])
            diff = np.setdiff1d(self.genes(), self[pathway])

            n_samples = np.minimum(len(diff), np.int(negpos_ratio*len(pathway)))

            negs = np.vstack((np.random.choice(diff, n_samples, replace=False),
                               np.array([pathway for _ in range(n_samples)])))

            negs = negs.transpose()

            X_train_pos, X_test_pos = getRandomSplit(pos, largest_fraction=train_ratio)
            Y_train_pos, Y_test_pos = np.ones(X_train_pos.shape[0]), np.ones(X_test_pos.shape[0])

            X_train += [X_train_pos]
            Y_train += [Y_train_pos]

            Y_test += [Y_test_pos]
            X_test += [X_test_pos]

            X_train_neg, X_test_neg = getRandomSplit(negs, largest_fraction=train_ratio)
            Y_train_neg, Y_test_neg = np.zeros(X_train_neg.shape[0]), np.zeros(X_test_neg.shape[0])

            X_train += [X_train_neg]
            Y_train += [Y_train_neg]

            Y_test += [Y_test_neg]
            X_test += [X_test_neg]

        return np.vstack(tuple(X_train)), np.vstack(tuple(X_test)), np.hstack(tuple(Y_train)), np.hstack(tuple(Y_test))


def checkInputGenelist(genelist):

    try:
        genelist = [s for s in genelist if isinstance(s, str) and (s != '')]

    except:
        raise IOError('Please provide an iterable as input.')

    try:
        genelist = np.array(genelist)

    except:
        raise IOError('Please check that the input array is well formatted.')

    genelist2 = np.unique(genelist)

    if len(genelist2) != len(genelist):
        warnings.warn('The provided list contains non-unique entries.')

    elif len(genelist2) == 0:
        raise IOError('Please provide a valid gene list.')

    return genelist2


def getRandomSplit(arr, largest_fraction=0.7):
    n_elements = arr.shape[0]
    large_ids = np.random.choice(n_elements, np.int(len(arr) * largest_fraction))
    smallest_ids = np.setdiff1d(np.arange(n_elements), large_ids)

    large_arr, small_arr = arr[large_ids], arr[smallest_ids]

    return large_arr, small_arr


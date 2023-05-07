import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from NetworkAnalysis.OmicsDataSet import OmicsDataSet
from joblib import Parallel, delayed

from lifelines import CoxPHFitter

# Test status: all functions have been tested
# TODO: apply changes for conversion to DF object
# Extend to include multiple states

class DiscreteOmicsDataSet(OmicsDataSet):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 patient_axis='auto', remove_nas=False, attrs=None, type='Omics',
                 remove_zv=False, verbose=True, average_dup_genes=False, average_dup_samples=False):

        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy,
                         patient_axis=patient_axis, remove_nas=remove_nas,
                         type=type, remove_zv=remove_zv, verbose=verbose,
                         average_dup_genes=average_dup_genes,
                         average_dup_samples=average_dup_samples)

        if attrs is None:
            unique_levels = pd.unique(self.df.values.flatten())
            self.attrs = {i: ' ' + self.type + str(i) for i in unique_levels}
        else:
            self.attrs = attrs

        self.df = self.df.astype(np.uint16)

    @classmethod
    def fromAnnotationFile(cls, file):
        annotations = pd.read_csv(file, header=0, index_col=0)
        pats, genes = pd.unique(np.array(list(annotations.index))), pd.unique(annotations.GENE)
        pat2ind, gene2ind = {pats[i]: i for i in range(len(pats))}, {genes[i]: i for i in range(len(genes))}

        mut_data = np.zeros((len(pats), len(genes)))
        row_ids, col_ids = np.array(list(map(lambda x: pat2ind[x], list(annotations.index)))),\
                          np.array(list(map(lambda x: gene2ind[x], annotations.GENE)))
        mut_data[row_ids, col_ids] = 1
        mut_data = pd.DataFrame(mut_data, index=pats, columns=genes)

        return cls(mut_data, type='MUT', attrs=(' 0', ' 1'), remove_nas=False,
                   average_dup_genes=False, average_dup_samples=False)

    def __getitem__(self, item):
        return DiscreteOmicsDataSet(self.df.iloc[item], type=self.type, remove_zv=False, patient_axis=0)

    def compareSampleProfiles(self, patient_list=None, sort=True):
        if patient_list is None:
            plot_patients = self.samples()
        else:
            plot_patients = list(set(patient_list).intersection(self.samples(as_set=True)))

        plot_df = self.df.loc[plot_patients].transpose()

        if sort:
            nrows, ncols = plot_df.shape

            plot_df = plot_df.append(plot_df.sum(axis=0), ignore_index=True)
            plot_df = plot_df.sort_values(by=nrows, axis=1, ascending=False)
            plot_df = plot_df.drop(nrows, axis=0)
            plot_df['Rowsum'] = np.matmul(plot_df.values, 2.**np.arange(0, -ncols, -1))
            plot_df.index = self.genes()

            plot_df = plot_df.sort_values(by='Rowsum', ascending=False)
            plot_df = plot_df.drop('Rowsum', axis=1)

        plt.figure()
        plt.pcolor(plot_df)
        plt.colorbar()
        plt.yticks(np.arange(0.5, len(plot_df.index) , 1), list(plot_df.index))

        plt.xticks(np.arange(0.5, len(plot_df.columns), 1), plot_df.columns)
        plt.xticks(rotation=45)
        plt.show()

    def getDataFrame(self):
        return pd.DataFrame(self.values, index=list(self.index), columns=list(self.columns.values))

    def compareGeneProfiles(self, gene_list=None, sort=True):
        '''
        :param gene_list: list of genes that are plotted in the heatmap
        :param sort: does the x axis (containing the genes) have to be sorted
        :return: a heatmap showing the genes in genelist across all samples
        '''
        if gene_list is None:
            plot_genes = self.genes()
        else:
            plot_genes = list(set(gene_list).intersection(self.genes(as_set=True)))

        plot_df = self.df[plot_genes]

        if sort:
            nrows, ncols = plot_df.shape

            plot_df = plot_df.append(plot_df.sum(axis=0), ignore_index=True)
            plot_df = plot_df.sort_values(by=nrows, axis=1, ascending=False)
            plot_df = plot_df.drop(nrows, axis=0)
            plot_df['Rowsum'] = np.matmul(plot_df.values, 2.**np.arange(0, -ncols, -1))
            plot_df.index = self.samples()

            plot_df = plot_df.sort_values(by='Rowsum', ascending=False)
            plot_df = plot_df.drop('Rowsum', axis=1)

        plt.figure()
        plt.pcolor(plot_df)
        plt.colorbar()
        plt.yticks(np.arange(0.5, len(plot_df.index), 1), list(plot_df.index))
        plt.xticks(np.arange(0.5, len(plot_df.columns), 1), plot_df.columns)
        plt.xticks(rotation=45)
        plt.show()

    def concatDatasets(self, datasets, axis=1, include_types=True, average_dup_genes=True, average_dup_samples=True):
        DF = super().concatDatasets(datasets, axis, include_types)

        return DiscreteOmicsDataSet(DF, patient_axis=0,
                                    average_dup_genes=average_dup_genes,
                                    average_dup_samples=average_dup_samples)

    def subsetGenes(self, genes, inplace=False):
        if not inplace:
            return DiscreteOmicsDataSet(super().subsetGenes(genes), patient_axis=0, type=self.type, attrs=self.attrs)
        else:
            super().subsetGenes(genes, inplace=True)

    def subsetSamples(self, samples, inplace=False):
        if not inplace:
            return DiscreteOmicsDataSet(super().subsetSamples(samples), patient_axis=0, type=self.type, attrs=self.attrs)
        else:
            super().subsetGenes(samples, inplace=True)

    def GetPatientDistances(self, norm):
        pass

    def GetGeneDistances(self, norm):
        pass

    def invert(self):
        self.df = 1 - self.df

    def getNonzeroSampleDict(self):
        '''
        :return: returns a dict containing the samples as keys and a numpy array of all nonzero genes as values
        '''
        self.removeZeroVarianceGenes()

        return self.df.apply(lambda x: self.genes()[x != 0], axis=0).to_dict()

    def compareAllPairs(self, count_thresh):
        '''
        :param count_thresh: a threshold, all regimes with less than this amount of observations are thrown out
        :return: a pval_df containing all significant pairs for all possible combinations
        '''
        pass

    def thresholdGenes(self, count_thresh=0.1, sides='right', inplace=False):

        new_df = thresholdGenes(self.df, count_thresh=count_thresh, sides=sides)

        if inplace:
            self.df = new_df
        else:
            return DiscreteOmicsDataSet(new_df, patient_axis=0, type=self.type, attrs=self.attrs)

    def calculateHazardRatios(self, event_labels, os, age=None, shuffle=False):
        if shuffle:
            samples = self.samples
            df = self.df.sample(frac=1)
            df.index = samples
        else:
            df = self.df

        if len(self.unique()) > 2:
            HR_data = [getHazardRatio((df[col] == value).astype(np.int), os, event_labels, col, value,
                                      age=age, return_sign=True)
                       for col in df for value in np.unique(df[col].values)]
        else:
            print('Binary data, only considering one regime per gene profile.')
            HR_data = [getHazardRatio((df[col] == value).astype(np.int), os, event_labels, col, value, age=age,
                                      binary=True, return_sign=True)
                       for col in df for value in np.unique(df[col].values)[:-1]]

        return pd.DataFrame(HR_data, columns=['Gene', 'Regime', 'Hazard Ratio', 'N_samples'])

    def batchSignificantGenePairs(self, dataset2=None, n_batches=1, testtype='right', count_thresh=20,
                                  pvals_thresh=np.log(1e-10), sort=True):

        self.thresholdGenes(count_thresh=count_thresh, sides='both', inplace=True)

        if dataset2 is not None:
            dataset2.thresholdGenes(count_thresh=count_thresh, sides='both', inplace=True)

            dataset2 = dataset2



        print('Starting the analysis')
        print('Dimension of dataset 1: %i samples and %i columns' % (self.df.shape))
        print('Dimension of dataset 2: %i samples and %i columns' % (dataset2.df.shape))

        pvals = batchDecorator(get_pval_two_mat, self.df, dataset2.df, n_batches=n_batches, testtype=testtype,
                               count_thresh=count_thresh, pvals_thresh=pvals_thresh)

        if sort:
            pvals = pvals.sort_values(by='p-value')

        return pvals

    def batchCondProb(self, dataset2=None, n_batches=1, count_thresh=20, prob_thresh=0.5, sort=True):
        if dataset2 is not None:
            dataset2 = dataset2.df

        probs = batchDecorator(get_cond_probs, self.df, dataset2, n_batches=n_batches,
                               count_thresh=count_thresh, prob_thresh=prob_thresh)

        if sort:
            probs = probs.sort_values(by='Prob')

        return probs

    def getSignificantGenePairs(self, dataset2=None, testtype='right', count_thresh=20,
                                pvals_thresh=1e-10, garbage_cluster=None):

        self_vals = np.unique(self.df.values)
        if garbage_cluster is not None:
            self_vals = self_vals[self_vals != garbage_cluster]
        N_vals = len(self_vals)

        if (dataset2 is None) or self.equals(dataset2):
            pval_df = [get_pval_two_mat((self.df == self_vals[v1]).astype(np.uint16),
                                        (self.df == self_vals[v2]).astype(np.uint16),
                                        pvals_thresh=pvals_thresh, count_thresh=count_thresh,
                                        testtype=testtype, attr=(self.attrs[self_vals[v1]], self.attrs[self_vals[v2]]))
                       for v1 in range(N_vals) for v2 in range(v1 + 1)]

            pval_df = pd.concat(pval_df, axis=0)
            pval_df = pval_df.sort_values(by='p-value')
            return pval_df

        else:
            other_vals = dataset2.unique()
            if garbage_cluster is not None:
                other_vals = other_vals[other_vals != garbage_cluster]

            pval_df = [get_pval_two_mat((self.df == v1).astype(np.uint16),
                                        (dataset2.df == v2).astype(np.uint16),
                                        pvals_thresh=pvals_thresh, count_thresh=count_thresh,
                                        testtype=testtype, attr=(self.attrs[v1], dataset2.attrs[v2]))
                       for v1 in self_vals for v2 in other_vals]

            pval_df = pd.concat(pval_df, axis=0)
            pval_df = pval_df.sort_values(by='p-value')

            return pval_df

    def getConditionalProbabilities(self, dataset2=None, count_thresh=20, prob_thresh=0.95, attr=None):
        self_vals = np.unique(self.df.values)
        N_vals = len(self_vals)

        if (dataset2 is None) or self.equals(dataset2):
            pval_df = [get_cond_probs((self.df == self_vals[v1]).astype(np.uint16),
                                        (self.df == self_vals[v2]).astype(np.uint16),
                                        prob_thresh=prob_thresh, count_thresh=count_thresh,
                                        attr=(self.attrs[self_vals[v1]], self.attrs[self_vals[v1]]))
                      for v1 in range(N_vals) for v2 in range(v1+1)]

            pval_df = pd.concat(pval_df, axis=0)
            pval_df = pval_df.sort_values(by='Prob')
            return pval_df

        else:
            other_vals = dataset2.unique()

            pval_df = [get_cond_probs((self.df == v1).astype(np.uint16),
                                      (dataset2.df == v2).astype(np.uint16),
                                      prob_thresh=prob_thresh, count_thresh=count_thresh,
                                      attr=(self.attrs[v1], dataset2.attrs[v2]))
                       for v1 in self_vals for v2 in other_vals]

            pval_df = pd.concat(pval_df, axis=0)
            pval_df = pval_df.sort_values(by='Prob')

        return pval_df

    def new_getSignificantGenePairs(self, dataset2=None, count_thresh=20, pvals_thresh=1e-10,
                                    garbage_clusters=[], include_table=False):
        '''
        Calculate contingency tables and use those to calculate p-values
        :param dataset2:
        :param count_thresh:
        :param pvals_thresh:
        :param garbage_clusters: clusters that should be excluded when calculating p-values.
        '''
        # ToDo: calculate only some of the tables and fill in the rest based on totals

        self_vals = np.unique(self.df.values)
        self_vals = [val for val in self_vals if val not in garbage_clusters]
        N_vals = len(self_vals)

        if dataset2 is None:
            # ToDo: make this more efficient
            dataset2 = self
        other_vals = dataset2.unique()
        other_vals = [val for val in other_vals if val not in garbage_clusters]
        count_df = [get_counts((self.df == v1).astype(np.uint16),
                               (dataset2.df == v2).astype(np.uint16),
                               count_thresh=0, attr=(self.attrs[v1], dataset2.attrs[v2]))
                    for v1 in self_vals for v2 in other_vals]

        genes = count_df[0].iloc[:, :2]

        associations = pd.MultiIndex.from_tuples([(self.attrs[v1],
                                                   dataset2.attrs[v2]) for v1 in self_vals for v2 in other_vals])
        counts = [df['Count'] for df in count_df]
        count_df = pd.DataFrame(counts, index=associations).transpose()

        # For each row, calculate the p-value
        pval_df = [pval_from_cont_table(count_df, genes, (self.attrs[v1], dataset2.attrs[v2]),
                                        count_thresh=count_thresh, pvals_thresh=pvals_thresh)
                   for v1 in self_vals for v2 in other_vals]
        pval_df = pd.concat(pval_df, axis=0)
        pval_df = pval_df.sort_values(by='p-value')

        if include_table:
            associations = [self.attrs[v1] + dataset2.attrs[v2] for v1 in self_vals for v2 in other_vals]
            count_df = pd.DataFrame(counts, index=associations).transpose()
            count_df = pd.concat([genes, count_df], axis=1)
            pval_df = pd.merge(pval_df, count_df, how='left', on=['Gene_A', 'Gene_B'])
            return pval_df
        else:
            return pval_df

    def getPMI(self, dataset2=None, count_thresh=20, prob_thresh=0.95, attr=None):
        pass
    #TODO implement this


def get_counts(df1, df2, count_thresh=0, attr=None):

    if len(df1.shape) == 1:
        df1 = pd.DataFrame({'Query': df1, 'Dummy': [0 for i in range(df1.shape[0])]})
    elif len(df2.shape) == 1:
        df2 = pd.DataFrame({'Query': df2, 'Dummy': [0 for i in range(df2.shape[0])]})

    gene_ids1 = df1.columns.values
    gene_ids2 = df2.columns.values

    cooc_mat = df1.transpose().dot(df2).values
    if df1.equals(df2):
        cooc_mat = np.triu(cooc_mat, 1)

    ids = np.where(cooc_mat >= count_thresh)
    cooc_mat = cooc_mat[ids]
    gene_ids1, gene_ids2 = gene_ids1[ids[0]], gene_ids2[ids[1]]

    if attr is not None:
        count_df = pd.DataFrame({'Gene_A': gene_ids1, 'Gene_B': gene_ids2, 'Regime_A': attr[0],
                                'Regime_B': attr[1], 'Count': cooc_mat})
    else:
        count_df = pd.DataFrame({'Gene_A': gene_ids1, 'Gene_B': gene_ids2, 'Count': cooc_mat})

    return count_df


def pval_from_cont_table(count_df, genes, attr, pvals_thresh=1e-10, count_thresh=20):
    '''
    :param count_df: dataframe containing counts per regime combination
    :param genes: a dataframe with two columns, indicating the genes that are compared
    :param attr: a tuple of strings, used to indicate the regime of the data
    :param pvals_thresh: float consider only pairs that have a p-value below this threshold
    :param count_thresh: integer, only co-occurrence above this threshold is kept
    :return: a pval df containing the most significant pairs and their p-values
    '''
    v1, v2 = attr
    k = count_df[v1, v2]

    mask = np.where(k > count_thresh)
    count_df = count_df.loc[mask]
    genes = genes.loc[mask]
    k = k[mask[0]]

    n = count_df.sum(axis=1)
    p = count_df[v1].sum(axis=1) * count_df.loc[:, pd.IndexSlice[:, v2]].sum(axis=1) / n ** 2
    pvals = binom.sf(k, n, p)

    pval_df = pd.DataFrame({'Gene_A': genes['Gene_A'], 'Gene_B': genes['Gene_B'], 'Regime_A': v1,
                            'Regime_B': v2, 'Count': k, 'p-value': pvals})
    pval_df = pval_df.loc[pval_df['p-value'] < pvals_thresh, :]

    return pval_df


def get_pval_one_mat(df1, testtype='right', pvals_thresh=1e-10, count_thresh=20, attr=None):
    '''
    :param df1: the dataframe containing the binary data
    :param testtype: whether left- or rightsided tests needs to be done
    :param pvals_thresh: float consider only pairs that have a p-value below this threshold
    :param count_thresh: integer, only co-occurrence above this threshold is kept
    :param attr: a tuple of strings, can be used to indicate the regime of the data
    :return: a pval df containing the most significant pairs and their p-values
    '''

    df1 = df1[df1.columns.values[df1.sum(axis=0) > count_thresh]]
    gene_ids = df1.columns.values
    cooc_mat = df1.transpose().dot(df1).values
    P_mat = np.outer(df1.sum(axis=0).values, df1.sum(axis=0).values)
    cooc_mat = np.triu(cooc_mat, 1)
    ids = np.where(cooc_mat > count_thresh)

    gene_as, gene_bs = gene_ids[ids[0]], gene_ids[ids[1]]
    P_mat, cooc_mat = P_mat[ids], cooc_mat[ids]

    N_samples = df1.shape[0]

    if testtype.lower() == 'left':
        pvals_mat = binom.logcdf(cooc_mat, N_samples, 1. * P_mat/N_samples**2)
    else:
        pvals_mat = binom.logsf(cooc_mat, N_samples, 1. * P_mat/N_samples**2)

    pvals_mat[np.isinf(pvals_mat)] = -500
    mask = pvals_mat < pvals_thresh
    gene_as, gene_bs = gene_as[mask], gene_bs[mask]
    counts = cooc_mat[mask]
    pvals = pvals_mat[mask]

    if attr is not None:
        gene_as = np.core.defchararray.add(gene_as.astype('str'), attr[0])
        gene_bs = np.core.defchararray.add(gene_bs.astype('str'), attr[1])

    pval_df = pd.DataFrame({'Gene_A': gene_as, 'Gene_B': gene_bs, 'Count': counts, 'p-value': pvals})
    pval_df.sort_values(by='p-value', inplace=True)
    return pval_df


def get_pval_two_mat(df1, df2, testtype='right', pvals_thresh=1e-10, count_thresh=20, attr=None):
    '''
    :param df1: the dataframe containing the binary data
    :param testtype: whether left- or rightsided tests needs to be done
    :param pvals_thresh: float consider only pairs that have a p-value below this threshold
    :param count_thresh: integer, only co-occurrence above this threshold is kept
    :param attr: a tuple of strings, can be used to indicate the regime of the data
    :return: a pval df containing the most significant pairs and their p-values
    '''

    N_samples = df2.shape[0]

    df1 = thresholdGenes(df1, count_thresh=count_thresh, sides=testtype)
    df2 = thresholdGenes(df2, count_thresh=count_thresh, sides=testtype)

    if len(df1.shape) == 1:
        df1 = pd.DataFrame({'Query': df1, 'Dummy': [0 for i in range(df1.shape[0])]})
    elif len(df2.shape) == 1:
        df2 = pd.DataFrame({'Query': df2, 'Dummy': [0 for i in range(df2.shape[0])]})

    gene_ids1 = df1.columns.values
    gene_ids2 = df2.columns.values

    cooc_mat = df1.transpose().dot(df2).values
    if df1.equals(df2):
        cooc_mat = np.triu(cooc_mat, 1)

    P_mat = np.outer(df1.sum(axis=0).values, df2.sum(axis=0).values)

    if testtype.lower() == 'left':
        ids = np.where(cooc_mat < count_thresh)
        cooc_mat, P_mat = cooc_mat[ids], P_mat[ids]
        gene_ids1, gene_ids2 = gene_ids1[ids[0]], gene_ids2[ids[1]]
        pvals_mat = binom.cdf(cooc_mat, N_samples, P_mat/N_samples**2)
    else:
        ids = np.where(cooc_mat > count_thresh)
        cooc_mat, P_mat = cooc_mat[ids], P_mat[ids]
        gene_ids1, gene_ids2 = gene_ids1[ids[0]], gene_ids2[ids[1]]
        pvals_mat = binom.sf(cooc_mat, N_samples, P_mat/N_samples**2)

    pvals_mat[np.isinf(pvals_mat)] = -500
    mask = np.where(pvals_mat < pvals_thresh)
    gene_ids1 = gene_ids1[mask]
    gene_ids2 = gene_ids2[mask]
    counts = cooc_mat[mask]
    pvals = pvals_mat[mask]

    if attr is not None:
        pval_df = pd.DataFrame({'Gene_A': gene_ids1, 'Gene_B': gene_ids2, 'Regime_A': attr[0],
                                'Regime_B': attr[1], 'Count': counts, 'p-value': pvals})
    else:
        pval_df = pd.DataFrame({'Gene_A': gene_ids1, 'Gene_B': gene_ids2, 'Count': counts, 'p-value': pvals})

    return pval_df


def get_cond_probs(df1, df2, prob_thresh=0.95, count_thresh=20, attr=None):

    if len(df1.shape) == 1:
        df1 = pd.DataFrame({'Query': df1, 'Dummy': [0 for _ in range(df1.shape[0])]})
    elif len(df2.shape) == 1:
        df2 = pd.DataFrame({'Query': df2, 'Dummy': [0 for _ in range(df2.shape[0])]})

    gene_ids1 = df1.columns.values
    gene_ids2 = df2.columns.values

    cooc_mat = df1.transpose().dot(df2).values
    P_mat = np.tile(df2.sum(axis=0).values, (cooc_mat.shape[0], 1))
    np.fill_diagonal(cooc_mat, 0)

    ids = np.where(cooc_mat > count_thresh)
    cooc_mat, P_mat = cooc_mat[ids], P_mat[ids]
    gene_ids1, gene_ids2 = gene_ids1[ids[0]], gene_ids2[ids[1]]
    prob_mat = cooc_mat/P_mat

    mask = np.where(prob_mat > prob_thresh)
    gene_ids1 = gene_ids1[mask]
    gene_ids2 = gene_ids2[mask]
    counts = cooc_mat[mask]
    probs = prob_mat[mask]

    if attr is not None:  # P(A|B), so gene A is the source and gene B is the target
        prob_df = pd.DataFrame({'Gene_A': gene_ids1, 'Gene_B': gene_ids2, 'Regime_A': attr[0],
                                'Regime_B': attr[1], 'Count': counts, 'Prob': probs})
    else:
        prob_df = pd.DataFrame({'Gene_A': gene_ids1, 'Gene_B': gene_ids2, 'Count': counts, 'Prob': probs})

    return prob_df


def getPMI(X1, X2, thresh=None):

    if isinstance(X1, pd.DataFrame):

        features1 = X1.columns.values
        X1 = X1.values

    if isinstance(X2, pd.DataFrame):

        features2 = X2.columns.values
        X2 = X2.values

    N = X1.shape[0]
    overlap = np.matmul(np.transpose(X1), X2)
    p_X1, p_X2 = np.sum(X1, axis=0), np.sum(X2, axis=0)

    PMI = np.log(N * overlap / np.outer(p_X1, p_X2) + 1e-15)

    if thresh is not None:
        row, col = np.where(np.abs(PMI) > thresh)

        return pd.DataFrame({'Gene_A': features1[row], 'Gene_B': features2[col], 'PMI': PMI[(row, col)]})

    else:

        return PMI, overlap / N


def getMI(X1, X2, thresh=None):

    N_features1, N_features2 = X1.shape[1], X2.shape[1]
    features1, features2 = np.arange(N_features1), np.arange(N_features2)

    if isinstance(X1, pd.DataFrame):

        features1 = X1.columns.values
        X1 = X1.values

    if isinstance(X2, pd.DataFrame):

        features2 = X2.columns.values
        X2 = X2.values

    N_reg1, N_reg2 = np.unique(X1), np.unique(X2)
    MI = np.zeros((N_features1, N_features2))

    for v1 in N_reg1:
        for v2 in N_reg2:
            PMI, overlap = getPMI(1 * (X1 == v1), 1 * (X2 == v2))
            MI += overlap * PMI

    if thresh is not None:
        row, col = np.where(MI > thresh)

        return pd.DataFrame({'Gene_A': features1[row], 'Gene_B': features2[col], 'MI': MI[(row, col)]})

    else:
        return MI


def getHazardRatio(df_col, os, event, genename, value, binary=False, age=None, return_sign=False):
    cph = CoxPHFitter()
    os_data = pd.DataFrame({'Gene': df_col,
                            'Duration': os,
                            'Flag': event})
    if age is not None:
        os_data['Age'] = age

    try:
        cph.fit(os_data, 'Duration', 'Flag', show_progress=False)
    except ValueError:
        print('Not working, returning nans')
        return genename, value, np.nan, df_col.sum()

    hazard_ratio = np.exp(cph.hazards_['Gene'].values)

    if binary:
        if hazard_ratio < 1:
            hazard_ratio = 1/hazard_ratio
            value = 1

    if return_sign:
        return genename, value, hazard_ratio[0], df_col.sum()
    else:
        return hazard_ratio


def batchDecorator(func, df1, df2=None, n_batches=1, **kwds):
    if df2 is None:
        n_cols = df1.shape[1]
        break_points = np.linspace(0, n_cols, num=n_batches+1, dtype=np.int)

        pvals = []
        df2 = df1

        for i in range(len(break_points) - 1):
            for j in range(i + 1):
                pvals_ = getAllRegimes(func, df1.iloc[:, break_points[i]:break_points[i + 1]],
                                             df2.iloc[:, break_points[j]:break_points[j + 1]], **kwds)
                pvals += [pvals_]

    else:
        n_cols1, n_cols2 = df1.shape[1], df2.shape[1]
        break_points1 = np.linspace(0, n_cols1, num=n_batches+1, dtype=np.int)
        break_points2 = np.linspace(0, n_cols2, num=n_batches+1, dtype=np.int)

        pvals = []

        for i in range(len(break_points1) - 1):
            for j in range(len(break_points2) - 1):
                pvals_ = getAllRegimes(func, df1.iloc[:, break_points1[i]:break_points1[i + 1]],
                                             df2.iloc[:, break_points2[j]:break_points2[j + 1]], **kwds)
                pvals += [pvals_]

    pvals = pd.concat(pvals, axis=0)

    return pvals


def getAllRegimes(func, df1, df2=None, **kwds):

    self_vals = np.unique(df1.values)
    N_vals = len(self_vals)

    if (df2 is None) or df1.equals(df2):
        pval_df = [func((df1 == self_vals[v1]).astype(np.uint16),
                        (df2 == self_vals[v2]).astype(np.uint16),
                        attr=(self_vals[v1], self_vals[v2]),
                        **kwds)
                  for v1 in range(N_vals) for v2 in range(v1+1)]

        pval_df = pd.concat(pval_df, axis=0)

        return pval_df

    else:
        other_vals = np.unique(df2.values)

        pval_df = [func((df1 == v1).astype(np.uint16),
                        (df2 == v2).astype(np.uint16),
                        attr=(v1, v2), **kwds)
                  for v1 in self_vals for v2 in other_vals]

        pval_df = pd.concat(pval_df, axis=0)

        return pval_df


def batchDecorator_new(func, df1, df2=None, n_batches=1, **kwds):
    if df2 is None:
        n_cols = df1.shape[1]
        break_points = np.linspace(0, n_cols, num=n_batches+1, dtype=np.int)

        pvals = []
        df2 = df1

        for i in range(len(break_points) - 1):
            for j in range(i + 1):
                pvals_ = func(df1.iloc[:, break_points[i]:break_points[i + 1]],
                              df2.iloc[:, break_points[j]:break_points[j + 1]], **kwds)
                pvals += [pvals_]

    else:
        n_cols1, n_cols2 = df1.shape[1], df2.shape[1]
        break_points1 = np.linspace(0, n_cols1, num=n_batches+1, dtype=np.int)
        break_points2 = np.linspace(0, n_cols2, num=n_batches+1, dtype=np.int)

        pvals = []

        for i in range(len(break_points1) - 1):
            for j in range(len(break_points2) - 1):
                pvals_ = func(df1.iloc[:, break_points1[i]:break_points1[i + 1]],
                              df2.iloc[:, break_points2[j]:break_points2[j + 1]], **kwds)
                pvals += [pvals_]

    pvals = pd.concat(pvals, axis=0)

    return pvals


def simplifyBins(data, cutoff=.1, max_bins=np.inf, inplace=True):
    '''
       Reorganizes bins by putting all bins below the cutoff into a separate "garbage" bin
       :param data: 1-D Pandas Series
       :param cutoff: minimal amount of samples per bin
       :param max_bins: maximum amount of bins (excluding the garbage bin)
       :param inplace:
       :return:
    '''
    bin_sizes = data.value_counts(normalize=True)
    new_values = pd.Series([i for i in range(len(bin_sizes))], index=bin_sizes.index)
    large_bins = bin_sizes[bin_sizes >= cutoff].index
    if len(large_bins) > max_bins:
        large_bins = large_bins[:max_bins]
    small_bins = bin_sizes.loc[~bin_sizes.index.isin(large_bins)].index
    new_values.loc[small_bins] = 999
    if inplace:
        data.replace(new_values, inplace=True)
    else:
        return data.replace(new_values)


def applyBinSimplfication(data, cutoff=.1, max_bins=np.inf, remove_zv=True, inplace=True, parallel=False, n_jobs=2):
    if parallel:
        new_data = Parallel(n_jobs=n_jobs)(delayed(simplifyBins)(data.df.loc[:, g], cutoff=cutoff, max_bins=max_bins,
                                                                 inplace=False) for g in data.genes())
    else:
        new_data = [simplifyBins(data.df.loc[:, g], cutoff=cutoff, max_bins=max_bins,
                                 inplace=False) for g in data.genes()]
    new_data = pd.DataFrame(new_data, index=data.genes()).transpose()
    if inplace:
        data.df = new_data
    else:
        return DiscreteOmicsDataSet(new_data, type=data.type, patient_axis=0, remove_zv=remove_zv)


def thresholdGenes(data, count_thresh=0.1, sides='right'):
    '''
    removes genes with counts below the threshold.
    :param data:
    :param count_thresh:
    :param sides: can be right, left or both
    returns a dataframe
    '''
    if (count_thresh < 1) and (count_thresh >= 0):
        c_ = np.int(count_thresh * len(data.index))
    elif (count_thresh >= 1) and (count_thresh < len(data.index)):
        c_ = np.int(count_thresh)
    else:
        raise IOError('Please provide count_thresh either as a fraction or as the count number.')

    genesum = data.sum(axis=0)

    if sides.lower() == 'both':
        data = data[data.columns[(genesum > c_) & (genesum < (len(data.index) - c_))]]
    elif sides.lower() == 'right':
        data = data[data.columns[genesum > c_]]
    elif sides.lower() == 'left':
        data = data[data.columns[genesum < (len(data.index) - c_)]]
    else:
        raise IOError('sides can be either both, left or right')

    return data

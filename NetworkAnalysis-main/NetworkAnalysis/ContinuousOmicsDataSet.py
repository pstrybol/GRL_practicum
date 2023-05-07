import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from cycler import cycler
from scipy.stats import expon, pearsonr, spearmanr, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from NetworkAnalysis.DiscreteOmicsDataSet import DiscreteOmicsDataSet
from NetworkAnalysis.UndirectedInteractionNetwork import UndirectedInteractionNetwork
from NetworkAnalysis.OmicsDataSet import OmicsDataSet
import copy
import itertools
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from sklearn.model_selection import GridSearchCV
from bisect import bisect_left

# TODO: run test script again
# TODO: change test script
#

class ContinuousOmicsDataSet(OmicsDataSet):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 patient_axis='auto', remove_nas=True, type='Omics',
                 remove_zv=True, verbose=True,
                 average_dup_genes=True, average_dup_samples=True):

        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy,
                         patient_axis=patient_axis, remove_nas=remove_nas, type=type, remove_zv=remove_zv,
                         verbose=verbose, average_dup_genes=average_dup_genes,
                         average_dup_samples=average_dup_samples)

    def compareSampleProfiles(self, kind='density', randomseed=None, Npatients=4):
        np.random.seed(randomseed)

        random_patients = list(np.random.permutation(self.samples())[:Npatients])
        plot_df = self.df.loc[random_patients]
        plot_df = plot_df.transpose()
        if kind == 'density':
            plot_df.plot(kind='density', subplots=True, use_index=False)
        else:
            plot_df.hist(bins=100)
        plt.show()

    def compareGeneProfiles(self, gene_list=None, Ngenes=4, kind='histogram'):
        if gene_list is None:
            random_genes = list(np.random.permutation(self.genes())[:Ngenes])
            print(random_genes)
            plot_df = self.df[random_genes]
        else:
            plot_genes = list(set(gene_list).intersection(set(self.genes())))
            plot_df = self.df[plot_genes]

        if plot_df.shape[1] > 0:
            if kind == 'density':
                plot_df.plot(kind='density', subplots=True, use_index=False)
            else:
                plot_df.hist(bins=100)

            plt.xlabel('Normalized expression', fontsize=14)
            plt.show()
        else:
            print('None of the genes are found in the dataset...')

    def normalizeProfiles(self, method='Quantile', axis=1, inplace=True):
        '''
        :param method: defines the method by which the (expression) profiles are scaled.
        currently supported is standardscaling, quantile normalization and centering.
        :param axis: the axis along which to scale (0 scales the genes profiles, 1 scales the patient profiles)
        :param inplace:
        :return: None (self.df is normalized)
        '''
        if axis == 1:
            self.df = self.df.transpose()

        if method.lower() == 'standardscaling':
            norm_df = (self.df - self.mean(axis=0))/self.std(axis=0)
        elif method.lower() == 'quantile':
            rank_mean = self.df.stack().groupby(self.df.rank(method='first').stack().astype(int)).mean()
            norm_df = self.df.rank(method='min').stack().astype(int).map(rank_mean).unstack()
        elif method.lower() == 'center':
            norm_df = self.df - self.mean(axis=0)
        elif method.lower() == 'min-max':
            min_, max_ = self.df.min(axis=0), self.df.max(axis=0)
            norm_df = (self.df - min_)/(max_ - min_)
        else:
            raise Exception('NotImplementedError')

        if axis == 1:
            self.df = self.df.transpose()
            norm_df = norm_df.transpose()

        if inplace:
            self.df = norm_df
        else:
            return ContinuousOmicsDataSet(norm_df, type=self.type, patient_axis=0)
        #TODO: invent fantastic scaling algorithm

    def concatDatasets(self, datasets, axis=1, include_types=True, average_dup_genes=True, average_dup_samples=True):
        DF = super().concatDatasets(datasets, axis, include_types)

        return ContinuousOmicsDataSet(DF, patient_axis=0,
                                      average_dup_genes=average_dup_genes,
                                      average_dup_samples=average_dup_samples)

    def __getitem__(self, item):
        if item > len(self.samples()):
            raise IndexError
        return ContinuousOmicsDataSet(self.df.iloc[item], type=self.type, remove_zv=False, patient_axis=0)

    def GetPatientDistances(self, norm):
        pass

    def GetGeneDistances(self, norm):
        pass

    def getGeneCorrelations(self, gene_list=None, method='pearson', **kwargs):
        '''
        :param gene_list: genes for which to calculate the correlation
        :param method: method to use for calculating the correlations.
        Options are:
        pearson
        spearman
        kendall, ranked-pearson, ranked-spearman and ranked-kendall.
        The ranked- version applies a ranking correction that both genes are also amongst the higest ranked
        genes in each other's list of co-expressed genes.
        '''
        if gene_list is not None:
            corrs, pvals = self.subsetGenes(gene_list).getGeneCorrelations(method=method)
            return corrs, pvals
        else:
            corrs = calculateAssociationSelf(self.df, method=method.lower(), **kwargs)
            return corrs

        # TODO add other methods

    def subsetGenes(self, genes, inplace=False, verbose=True):
        if not inplace:
            return ContinuousOmicsDataSet(super().subsetGenes(genes), patient_axis=0, type=self.type, verbose=verbose)
        else:
            super().subsetGenes(genes, inplace=True)

    def subsetSamples(self, samples, inplace=False):
        if not inplace:
            return ContinuousOmicsDataSet(super().subsetSamples(samples), patient_axis=0, type=self.type)
        else:
            super().subsetGenes(samples, inplace=True)

    def applyDensityBinarization(self, save_path=None, min_clustersize=150,
                                 max_nclusters=1000, p_thresh=1.e-5, MA_window=1,
                                 remove_zv=True):

        bin_gx = self.df.apply(lambda x: densityDiscretization1D(x, min_clustersize=min_clustersize,
                                                                 max_nclusters=max_nclusters, p_thresh=p_thresh,
                                                                 MA_window=MA_window), axis=0)

        if save_path is not None:
            print('data is being saved to:' + save_path)
            bin_gx.to_csv(save_path, sep='\t', index=True, header=True)

        return DiscreteOmicsDataSet(bin_gx, type=self.type, patient_axis=0, remove_zv=remove_zv)

    def applyGMMBinarization(self, save_path=None, max_regimes=2, remove_zv=False, criterion='bic', return_GMM=False):
        '''
        :param save_path: the path to which to save the binarized dataframe
        :param max_regimes: the max number of regimes
        :return: a DiscreteOmicsDataSet containing the discretized data
        '''

        np.random.seed(42)

        bin_gx = np.zeros(self.df.shape, dtype=np.uint16)
        GMMs = []
        id = 0

        for gene in self.genes():
            temp = self.df[gene]

            temp = np.reshape(temp.values, (-1, 1))
            max_val = np.max(temp, keepdims=True)
            print(id)
            gm_best, BIC_min, n_regimes = get_optimal_regimes(temp, max_regimes=max_regimes,
                                                              criterion=criterion)

            GMMs.append(gm_best)

            if n_regimes == 2:
                labels = gm_best.predict(temp)
                #labels = 1*(gm_best.predict(max_val) == labels)
                bin_gx[:, id] = labels.astype(np.uint16)
            else:
                labels = gm_best.predict(temp)
                bin_gx[:, id] = labels.astype(np.uint16)

            id += 1

        bin_gx = pd.DataFrame(bin_gx, index=self.samples(), columns=self.genes())
        GMMs = pd.Series(GMMs, index=self.genes())

        if save_path is not None:
            print('data is being saved to:' + save_path)
            bin_gx.to_csv(save_path, sep='\t', index=True, header=True)

        if return_GMM:
            return DiscreteOmicsDataSet(bin_gx, type=self.type, patient_axis=0, remove_zv=remove_zv), GMMs
        else:
            return DiscreteOmicsDataSet(bin_gx, type=self.type, patient_axis=0, remove_zv=remove_zv)

    def applyGMMBinarization_new(self, save_path=None, max_regimes=2, remove_zv=True, criterion='bic'):
        '''
        :param save_path: the path to which to save the binarized dataframe
        :param max_regimes: the max number of regimes
        :return: a DiscreteOmicsDataSet containing the discretized data
        '''
        np.random.seed(42)

        bin_gx = self.df.apply(lambda x: getGMMRegimes(x, max_regimes=max_regimes, criterion=criterion), axis=0)

        if save_path is not None:
            print('data is being saved to:' + save_path)
            bin_gx.to_csv(save_path, sep='\t', index=True, header=True)

        return DiscreteOmicsDataSet(bin_gx, type=self.type, patient_axis=0, remove_zv=remove_zv)

    def KmeansBinning(self, n_clusters=2, save_path=None, remove_zv=True):
        km = KMeans(n_clusters=n_clusters)
        km_bin_df = self.df.apply(lambda x: km.fit_predict(np.reshape(x, (-1, 1))), axis=0)

        if save_path is not None:
            print('data is being saved to:' + save_path)
            km_bin_df.to_csv(save_path, sep='\t', index=True, header=True)

        return DiscreteOmicsDataSet(km_bin_df, type=self.type, patient_axis=0, remove_zv=remove_zv)

    def BayesianGMM(self, n_clusters=2, save_path=None, remove_zv=True):
        km = BayesianGaussianMixture(n_clusters=n_clusters)
        km_bin_df = self.df.apply(lambda x: km.fit_predict(np.reshape(x, (-1, 1))), axis=0)

        if save_path is not None:
            print('data is being saved to:' + save_path)
            km_bin_df.to_csv(save_path, sep='\t', index=True, header=True)

        return DiscreteOmicsDataSet(km_bin_df, type=self.type, patient_axis=0, remove_zv=remove_zv)

    def applySTDBinning(self, std_thresh=1.5, save_path=None):

        scaled_df = (self.df - self.mean(axis=0))/self.std(axis=0)
        std_bin_gx = 1*(scaled_df > std_thresh) - 1*(scaled_df < - std_thresh)
        std_bin_gx = std_bin_gx.astype(np.uint16)
        std_bin_gx[~np.isfinite(std_bin_gx)] = 0

        if save_path is not None:
            print('data is being saved to:' + save_path)
            std_bin_gx.to_csv(save_path, sep='\t', index=True, header=True)

        return DiscreteOmicsDataSet(std_bin_gx, type=self.type, patient_axis=0)

    def applyMidRangedBinning(self):
        mid = (self.df.max(axis=0) - self.df.min(axis=0))/2
        binned_data = self.df > mid
        binned_data = binned_data.astype(np.uint16)
        return DiscreteOmicsDataSet(binned_data, type=self.type, patient_axis=0)

    def applyMaxMinusBinning(self, threshold=.5):
        cutoff = self.df.max(axis=0) - self.df.max(axis=0)*threshold
        binned_data = self.df > cutoff
        binned_data = binned_data.astype(np.uint16)
        return DiscreteOmicsDataSet(binned_data, type=self.type, patient_axis=0)

    def binData(self, method='GMM', n_regimes=3, remove_zv=False, threshold=.5, criterion='bic', optimize='NOT'):

        if 'kmeans' in method.lower():
            binned_data = self.KmeansBinning(n_clusters=n_regimes, remove_zv=remove_zv)
        elif 'gmm' in method.lower():
            binned_data = self.applyGMMBinarization(max_regimes=n_regimes, remove_zv=remove_zv, criterion=criterion)
        elif 'density' in method.lower():
            binned_data = self.applyDensityBinarization(max_nclusters=n_regimes, remove_zv=remove_zv)
        elif 'midranged' in method.lower():
            binned_data = self.applyMidRangedBinning()
        elif 'maxminus' in method.lower():
            binned_data = self.applyMaxMinusBinning(threshold)
        else:
            binned_data = self.applySTDBinning()

        return binned_data

    def getCorrelationArray(self, goi, method='pearson', **kwargs):
        cor_array = []
        for g in self.df.columns:
            if g == goi:
                cor_array.append(0)
            else:
                pair = self.subsetGenes([g, goi])
                cor_array.append(pair.getGeneCorrelations(method=method, **kwargs).iloc[0, 1])
        cor_array = pd.Series(cor_array, index=self.df.columns)
        return cor_array

    def compareCoExpression(self, gene, dataset2=None, sample_list1=None, sample_list2=None, method='pearson',
                            compare_method='orderedlist_measure', alpha = 0.001, **kwargs):
        '''
        Compare the coexpression of a gene with other genes between two datasets.
        :param dataset2: the expression dataset to compare to
        :param sample_list1 and 2: instead of two datasets, compare samples within this dataset
        '''
        if dataset2 is not None:
            dataset1 = self
            # make sure the same genes are present
            dataset1, dataset2 = dataset1.getCommonGenes(dataset2)
        elif sample_list1 is not None and sample_list2 is not None:
            dataset1 = self.subsetSamples(sample_list1)
            dataset2 = self.subsetSamples(sample_list2)
        else:
            raise IOError('No dataset to compare to. Use either dataset2 or sample_list1 and 2')

        # only one row of the matrix is used.
        # Consider making a custom correlation method that only calculates all pairs.
        coexp_1 = dataset1.getCorrelationArray(gene, method=method, **kwargs)
        coexp_2 = dataset2.getCorrelationArray(gene, method=method, **kwargs)

        # remove Na's (usually caused by genes without expression
        nas = np.logical_or(coexp_1.isna(), coexp_2.isna())
        coexp_1 = coexp_1[~nas]
        coexp_2 = coexp_2[~nas]

        if compare_method.lower() == 'pearson':
            result = pearsonr(coexp_1, coexp_2)[0]
        elif compare_method.lower() == 'spearman':
            result = spearmanr(coexp_1, coexp_2).correlation
        elif compare_method.lower() == 'orderedlist_measure':
            result = adjusted_orderedlist_measure(coexp_1, coexp_2, alpha, ascending=False)

        return result

    def plotExpressionRegime(self, gene, insert_title=True, savepath=None, remove_frame=False, criterion='bic',
                             annotated_patients=None, annotation_labels=None, max_regimes=2, method='GMM',
                             optimize='NOT', return_data=False):

        plotmat = self.df[gene].values
        max_val = np.max(plotmat)
        mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                                                'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                                                'tab:olive', 'tab:cyan'])

        labels = self.subsetGenes(gene).binData(method=method, n_regimes=max_regimes, criterion=criterion,
                                                optimize=optimize)
        labels = labels.df.values.flatten()

        n_regimes = len(np.unique(labels))
        print("Number of regimes: %i" %n_regimes)

        if n_regimes == 1:

            plot_df = pd.DataFrame({'Expression': plotmat.flatten(), 'Label': labels}, index=self.samples())
            bin_seq = np.linspace(plot_df.Expression.min(), plot_df.Expression.max(), num=200)
            label = ['Basal']
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_df.groupby('Label').hist(ax=ax, bins=bin_seq, label=label)

            if annotated_patients is not None:
                print('Adding annotation lines')
                r = plot_df.loc[annotated_patients]
                r.Expression.hist(ax=ax, bins=bin_seq, label=annotation_labels)
                label.extend(annotation_labels)
                print(label)

            plt.legend(label, fontsize=16)

        elif n_regimes == 2:

            plot_df = pd.DataFrame({'Expression': plotmat.flatten(), 'Label': labels}, index=self.samples())
            plot_df.sort_values('Expression', ascending=True, inplace=True)

            labels = ['Regime 0', 'Regime 1']
            fig, ax = plt.subplots(figsize=(8, 6))
            bin_seq = np.linspace(plot_df.Expression.min(), plot_df.Expression.max(), num=200)
            plot_df.groupby('Label').hist(ax=ax, bins=bin_seq, label=labels)

            if annotated_patients is not None:
                print('Adding annotation lines')
                r = plot_df.loc[annotated_patients]
                r.Expression.hist(ax=ax, bins=bin_seq, label=annotation_labels)
                labels.extend(annotation_labels)
                print(labels)

            plt.legend(labels, fontsize=18)

        else:
            plot_df = pd.DataFrame({'Expression': plotmat.flatten(), 'Label': labels}, index=self.samples())
            plot_df.sort_values('Expression', ascending=True, inplace=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            bin_seq = np.linspace(plot_df.Expression.min(), plot_df.Expression.max(), num=200)
            plot_df.groupby('Label').hist(ax=ax, bins=bin_seq)

            if annotated_patients is not None:
                print('Adding annotation lines')
                r = plot_df.loc[annotated_patients]
                r.Expression.hist(ax=ax, bins=bin_seq)

        if insert_title:
            plt.title('Expression profile for ' + str(gene), fontsize=18)
        else:
            plt.title('')

        plt.xlabel('Normalized expression', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.grid(False)

        if remove_frame:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if savepath is not None:
            print('Saving figure to: ' + savepath)
            plt.savefig(savepath, dpi=1000, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        if return_data:
            return plot_df

    def univariateTestingBinLabels(self, labels, func, n_jobs=1, **kwargs):

        if isinstance(labels, pd.Series) | isinstance(labels, pd.DataFrame):
            samples = np.array(list(labels.index))

        else:
            assert len(labels) == len(self.samples()), 'Please make sure that the labels are the same size as the number of samples'

        unique_labels = np.unique(labels)
        groups = [self.df.loc[labels == label] for label in unique_labels]

        if n_jobs == 1:
            pvals = [func(*[g[gene].values for g in groups], **kwargs) for gene in self.genes()]

        else:
            pvals = Parallel(n_jobs=n_jobs)(delayed(func)(*[g[gene].values for g in groups], **kwargs) for gene in self.genes())

        return pd.Series(pvals, index=self.genes()).sort_values()

    def removeStrangeBins(self, binned_data, inplace=True):
        '''
        Find and remove genes where one regime has samples assigned to it from both sides of another regime.
        :return:
        '''
        keep_genes = []
        discard_genes = []
        for gene in binned_data.genes():
            exp = self.df[gene]
            binned = binned_data.df[gene]
            unique_bins = pd.unique(binned)
            if len(unique_bins) == 1:
                keep_genes.append(gene)
            else:
                for combi in itertools.combinations(unique_bins, r=2):
                    bin1 = exp[binned == combi[0]]
                    bin2 = exp[binned == combi[1]]
                    if np.all(bin1 < min(bin2)) or np.all(bin1 > max(bin2)):
                        keep_genes.append(gene)
        if inplace:
            self.subsetGenes(keep_genes, inplace=True)
        else:
            return self.subsetGenes(keep_genes, inplace=False)

    def binningToScores(self, dataset2, pval_thresh=5e-5):
        scores = dict()
        for g in self.genes():
            scores[g] = new_bin_and_score(g, self, dataset2, pval_thresh=pval_thresh)
        scores = pd.DataFrame(scores)
        return ContinuousOmicsDataSet(scores, type=self.type)

    def Benchmarkbinning(self, labels, nregimes=None, nsplits=5, test_ratio=0.3, save_path=None):
        '''
        :param labels: provides class labels for each of the methods
        :param params: a nested dict containing the parameters for each binning method
        :return:
        '''

        sss = StratifiedShuffleSplit(n_splits=nsplits, test_size=test_ratio, random_state=0)
        sss.get_n_splits(self.df, labels)

        if nregimes is None:
            binning_methods = ['STD', 'GMM']
            nregimes = {'STD': 3, 'GMM': 3}
        else:
            binning_methods = list(nregimes.keys())

        scores_train, scores_val = pd.DataFrame(0, index=np.arange(nsplits), columns=binning_methods + ['Continuous']),\
                                   pd.DataFrame(0, index=np.arange(nsplits), columns=binning_methods + ['Continuous'])
        split_id = 0
        n_trees = 1500

        for train_index, test_index in sss.split(self.df, labels):

            X_train, X_val = self.df.iloc[train_index], self.df.iloc[test_index]
            Y_train, Y_val = labels[train_index], labels[test_index]
            print(self.df.shape)
            rf = RandomForestClassifier(n_estimators=n_trees)
            rf.fit(X_train, Y_train)

            scores_train.loc[split_id, 'Continuous'] = accuracy_score(Y_train, rf.predict(X_train))
            scores_val.loc[split_id, 'Continuous'] = accuracy_score(Y_val, rf.predict(X_val))

            for binning_method in binning_methods:
                bin_data = self.binData(method=binning_method, n_regimes=nregimes[binning_method])  #ZV features are automatically removed
                print(bin_data.df.shape)
                X_train, X_val = bin_data.df.iloc[train_index], bin_data.df.iloc[test_index]

                rf = RandomForestClassifier(n_estimators=n_trees)
                rf.fit(X_train, Y_train)

                scores_train.loc[split_id, binning_method] = accuracy_score(Y_train, rf.predict(X_train))
                scores_val.loc[split_id, binning_method] = accuracy_score(Y_val, rf.predict(X_val))

            split_id += 1

        ax = scores_val.boxplot(boxprops={'linewidth': 2}, flierprops={'linewidth': 2},
                                medianprops={'linewidth': 2, 'color': 'darkgoldenrod'})
        plt.xticks(fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if save_path is not None:
            plt.savefig(save_path + 'boxplot_binning_benchmark.pdf', dpi=1000, bbox_inches="tight")
            plt.savefig(save_path + 'boxplot_binning_benchmark.png', dpi=1000, bbox_inches="tight")

            scores_train.to_csv(save_path + 'training_scoresGMM.csv', header=True, index=False)
            scores_val.to_csv(save_path + 'val_scoresGMM.csv',  header=True, index=False)

        plt.show()

    def benchmarkOtherDataset(self, dataset2, labels1, labels2, params=None, n_regimes=3, save_path=None):
        print(self.df.shape)
        print(dataset2.df.shape)
        if params is None:
            binning_methods = ['Kmeans', 'GMM', 'Density'] # make STD work!
        else:
            binning_methods = params.keys()

        scores_train, scores_val, methods = [], [], []
        n_trees = 1500

        bin_dict1 = {binning_method: self.binData(method=binning_method, n_regimes=n_regimes).df
                    for binning_method in binning_methods}

        bin_dict2 = {binning_method: dataset2.binData(method=binning_method, n_regimes=n_regimes).df
                    for binning_method in binning_methods}

        bin_dict1['Continuous'], bin_dict2['Continuous'] = copy.deepcopy(self.df), copy.deepcopy(dataset2.df)

        for method, dataset in bin_dict1.items():
            print('##########################' + method + '####################################')
            print(dataset.shape)

            test_data = bin_dict2[method]
            print(test_data.shape)

            if dataset.shape[1] != test_data.shape[1]:
                genes_train = set(dataset.columns.values)
                genes_test = set(test_data.columns.values)

                if len(genes_train) > len(genes_test):
                    genediff = genes_train.difference(genes_test)
                    print('Genes that are only present in the training data: ')
                    print(genediff)
                    print(dataset[list(genediff)])

                else:
                    genediff = genes_test.difference(genes_train)
                    print('Genes that are only present in the testing data: ')
                    print(genediff)
                    print(test_data[list(genediff)])

            if method.lower() == 'continuous':
                sd = StandardScaler()
                dataset = sd.fit_transform(dataset)

                test_data = sd.fit_transform(test_data)

            rf = RandomForestClassifier(n_estimators=n_trees)
            rf.fit(dataset, labels1)

            scores_train += [accuracy_score(labels1, rf.predict(dataset))]
            scores_val += [accuracy_score(labels2, rf.predict(test_data))]
            methods += [method]

        results_df = pd.DataFrame({'Training': scores_train,
                                   'Test': scores_val}, index=methods).plot.bar(rot=45)

        return scores_train, scores_val, methods

    def getPosNegPairs(self, dataset2=None, n_batches=1, method='pearson', pos_criterion='5%',
                       neg_criterion='5%', gamma=5, alpha=1):
        if dataset2 is None:
            df2 = None
        else:
            df2 = dataset2.df

        pos, neg = batchPairConstruction(self.df, df2=df2, n_batches=n_batches, method=method,
                                         pos_criterion=pos_criterion, neg_criterion=neg_criterion,
                                         gamma=gamma, alpha=alpha)
        return pos, neg

    def getTrainTestPairs(self, corr_measure='pearson', train_ratio=0.7, neg_pos_ratio=5,
                          sampling_method='balanced', return_summary=True, **kwds):
        '''
        :param corr_measure:
        :param train_ratio:
        :param neg_pos_ratio:
        :param sampling_method:
        :param return_summary:
        :param kwds:
        :return: a tuple containing X_train, X_test, Y_train, Y_test, and optionally a summary of the data
        '''
        pos, _ = batchPairConstruction(self.df, df2=None, method=corr_measure, **kwds)
        graph_ = UndirectedInteractionNetwork(pos, colnames=('Gene_A', 'Gene_B'))
        train_data = graph_.getTrainTestData(train_ratio=train_ratio,
                                             neg_pos_ratio=neg_pos_ratio,
                                             method=sampling_method,
                                             return_summary=return_summary)

        return train_data


def new_bin_and_score(gene, dataset1, dataset2, pval_thresh=5e-5):
    data1, data2 = dataset1.df.loc[:, gene], dataset2.df.loc[:, gene]
    labels, GMM = getGMMRegimes(data1.values, max_regimes=2, return_GMM=True)

    # For each bin, calculate MWU p-values
    pvals = []
    for label in np.unique(labels):
        pvals.append(mannwhitneyu(data1[labels == label], data2)[1])
    pvals = pd.Series(pvals, index=np.unique(labels))

    gx = pd.Series(abs((data1 - np.mean(data2)) / (np.std(data2) + 0.01)), index=data1.index)

    # If a bin is not significantly different from the normal data, set z-scores to 0
    for label in np.unique(labels):
        if mannwhitneyu(data1[labels == label], data2)[1] > pval_thresh:
            gx[labels == label] = 0

    gx[gx > 10] = 10

    return gx


def bin_and_score(gene, dataset1, dataset2, pval_thresh=5e-5):

    data1, data2 = dataset1.df.loc[:, gene], dataset2.df.loc[:, gene]
    labels, GMM = getGMMRegimes(data1.values, max_regimes=2, return_GMM=True)

    mus = GMM.means_[:, 0]
    variances = GMM.covariances_[:, 0, 0]
    sigmas = np.sqrt(variances)

    pvals = []
    for label in np.unique(labels):
        pvals.append(mannwhitneyu(data1[labels == label], data2)[1])
    pvals = pd.Series(pvals, index=np.unique(labels))

    pval_bools = pvals < pval_thresh

    gx = pd.Series(np.zeros(len(data1)), index=data1.index)
    if 0 < sum(pval_bools) < len(pval_bools):
        # determine which GMM is the one closest to the normal samples
        normal_bin = pvals[pvals == max(pvals)].index[0]
        gx = abs((data1 - mus[normal_bin]) / (sigmas[normal_bin] + 0.01))
        gx[labels == normal_bin] = 0
        gx[gx > 10] = 10
    gx.name = gene
    return gx

def On(G1, G2, n, ascending=True):
    """ Calculate the number of overlapping genes in the top n of two lists, G1 and G2"""
    if ascending:
        top_G1 = G1.nsmallest(n).index.tolist()
        top_G2 = G2.nsmallest(n).index.tolist()
    else:
        top_G1 = G1.nlargest(n).index.tolist()
        top_G2 = G2.nlargest(n).index.tolist()

    return len(set(top_G1).intersection(set(top_G2)))


def orderedlist_measure(ranklist_1, ranklist_2, alpha):
    """ Input: 2 ranked lists where the lowest value = most significant"""

    result = 0

    for n in range(1, len(ranklist_1) + 1):
        result += np.e ** (-1 * alpha * n) * On(ranklist_1, ranklist_2, n)

    return result


def adjusted_orderedlist_measure(ranklist_1, ranklist_2, alpha, ascending=True):
    """
    Input: 2 ranked lists where the lowest value = most significant.
    NB: The ranklists should contain the same genes!

    ascending: if true, the most correlated gene has the lowest score
    """

    S = 0
    Enull = 0
    maxS = 0

    for n in range(1, len(ranklist_1) + 1):
        weight = np.e ** (-1 * alpha * n)
        S += weight * On(ranklist_1, ranklist_2, n, ascending=ascending)
        Enull += weight * n ** 2 / len(ranklist_1)
        maxS += weight * n

    return (S - Enull) / (maxS - Enull)


def get_optimal_regimes(data1d, max_regimes=2, criterion='bic', penalty=5):
    BIC_min, n_regimes = 1.e20, 1
    gm_best = None

    for regimes in range(1, max_regimes+1):
        gm = GaussianMixture(n_components=regimes, random_state=0) #42
        gm.fit(data1d)
        if criterion.lower() == 'aic':
            bic = gm.aic(data1d)
        elif criterion.lower() == 'rbic':
            bic = rbic(gm, data1d, penalty=penalty)
        else:
            bic = gm.bic(data1d)

        if bic < BIC_min:
            gm_best = gm
            BIC_min = bic
            n_regimes = regimes

    return gm_best, BIC_min, n_regimes

def getGMMRegimes(v, max_regimes, criterion='bic', return_GMM=False):
    temp = np.reshape(v, (-1, 1))
    max_val = np.max(temp)
    gm_best, BIC_min, n_regimes = get_optimal_regimes(temp, max_regimes=max_regimes, criterion=criterion)

    if n_regimes == 2:
        labels = gm_best.predict(temp)
        #labels = 1*(gm_best.predict(max_val) == labels)
        v_out = labels.astype(np.uint16)
    else:
        labels = gm_best.predict(temp)
        v_out = labels.astype(np.uint16)

    if return_GMM:
        return v_out, gm_best
    return v_out

def densityDiscretization1D(v, min_clustersize=200, max_nclusters=1000, p_thresh=0.0001, MA_window=1):

    # step 1: calculate distances
    v_sort = np.sort(v)
    v_diff = v_sort[1:] - v_sort[:-1]

    if MA_window > 1:
        window_size = list(np.arange(1, MA_window)) + [MA_window] * len(v_diff) + list(np.arange(MA_window-1, 0, -1))
        ids_begin = [0] * (MA_window - 1) + list(np.arange(len(v_diff))) + [len(v_diff)-1] * (MA_window -1)
        v_diff = np.array([np.sum(v_diff[ids_begin[i]:(window_size[i]+ids_begin[i])])/window_size[i] for i in range(len(ids_begin))])

    pvals = expon.sf(np.max(v_diff), scale=np.mean(v_diff[v_diff < np.percentile(v_diff, q=95)]))

    # step 2: select the areas with the lowest densities
    nclusters = 1
    min_ = np.min(v_diff)
    v_diff[:min_clustersize] = min_
    v_diff[-min_clustersize:] = min_
    v_diff[pvals > p_thresh] = min_
    break_points = []

    while (np.sum(v_diff > min_) > 0) & (nclusters < max_nclusters):
        id_breakpoint = np.argmax(v_diff)
        id_low, id_high = np.minimum(0, id_breakpoint), np.maximum(len(v_diff), id_breakpoint)
        v_diff[id_low:id_high] = min_
        nclusters += 1
        break_points += [v_sort[id_breakpoint]]

    if len(break_points) > 0:
        labels = np.sum(np.array([1 * (break_point < v) for break_point in break_points]), axis=0)
    else:
        labels = np.array([0 for _ in range(len(v))])

    return labels

def rbic(GMMobject, X, penalty=5):
    """Bayesian information criterion for the current model on the input X.
    Parameters
    ----------
    X : array of shape (n_samples, n_dimensions)
    Returns
    -------
    bic : float
        The lower the better.
    """
    return (-2 * GMMobject.score(X) * X.shape[0] +
            penalty * GMMobject._n_parameters() * np.log(X.shape[0]))

def calculateAssociationOther(df, df2=None, method='pearson', alpha=1, gamma=5, as_pairs=False,
                             pos_criterion='5%', neg_criterion='5%'):
        pass

def calculateAssociationSelf(df, method='pearson', alpha=1, gamma=5, as_pairs=False,
                             pos_criterion='5%', neg_criterion='5%'):

    genes_a = df.columns.values

    if method.lower() == 'pearson':
        dist = df.corr(method='pearson').abs()

    elif method.lower() == 'spearman':
        dist = df.corr(method='spearman').abs()

    elif method.lower() == 'kendall':
        dist = df.corr(method='kendall').abs()

    elif method.lower() == 'ranked-pearson':
        dist = df.corr(method='pearson').abs()
        dist = np.argsort(np.absolute(dist), axis=0)
        dist = np.exp(-(np.sqrt(np.transpose(dist) * dist) - alpha)/gamma)

    elif method.lower() == 'ranked-spearman':
        dist = df.corr(method='spearman').abs()
        dist = np.argsort(np.absolute(dist), axis=0)
        dist = np.exp(-(np.sqrt(np.transpose(dist) * dist) - alpha)/gamma)

    elif method.lower() == 'ranked-kendall':
        dist = df.corr(method='kendall').abs()
        dist = np.argsort(np.absolute(dist), axis=0)
        dist = np.exp(-(np.sqrt(np.transpose(dist) * dist) - alpha)/gamma)

    else:
        dist = 0
        raise IOError('The provided method is not known, possible methods are:'
                      '\'pearson\', \'spearman\', \'kendall\','
                      ' \'ranked-pearson\', \'ranked-spearman\', \'ranked-kendall\'.')

    dist = pd.DataFrame(dist, index=genes_a, columns=genes_a)

    if as_pairs:
        pos_pairs, neg_pairs = createPairsFromDistance(dist, upper_criterion=pos_criterion,
                                                       lower_criterion=neg_criterion)
        return pos_pairs, neg_pairs
    else:
        return dist


def checkCriterion(dist_df, criterion):

    N = dist_df.shape[0] * dist_df.shape[1]
    if (isinstance(criterion, str)) and ('%' in criterion):
        criterion = np.float(criterion.split('%')[0])

        if (criterion >= 0) and (criterion <= 100):
            th = np.percentile(dist_df, np.int(criterion), interpolation='lower')
        else:
            raise IOError('Percentages are between 0 and 100.')

    elif (np.float(criterion) > 0) and (np.float(criterion) < 1):
        # assumes the measure is scaled between [0, 1]
        th = criterion

    elif (np.float(criterion) > 0) and (np.float(criterion) < N):
        flat_dist = dist_df.values.flatten()
        th = flat_dist[np.argpartition(flat_dist, np.int(criterion))[np.int(criterion)]]

    else:
        raise IOError('Please provide a valid threshold as input.')

    print(th)

    return th


def createPairsFromDistance(dist_df, upper_criterion='95%', lower_criterion='5%'):
    gene_ids1 = np.array(list(dist_df.index))
    gene_ids2 = dist_df.columns.values

    th_up = checkCriterion(dist_df, upper_criterion)
    th_down = checkCriterion(dist_df, lower_criterion)

    dist_df = dist_df.values

    if np.allclose(dist_df.T, dist_df):
        r, c = np.triu_indices(dist_df.shape[0], k=1)
        dist_df = dist_df[(r, c)]
        genes1 = gene_ids1[r]
        genes2 = gene_ids2[c]

        ids = np.where(dist_df > th_up)
        gene1_pos = genes1[ids]
        gene2_pos = genes2[ids]
        association = dist_df[ids]

        pos_df = pd.DataFrame({'Gene_A': gene1_pos, 'Gene_B': gene2_pos, 'Association Strength': association})

        ids = np.where(dist_df < th_down)
        gene1_neg = genes1[ids]
        gene2_neg = genes2[ids]
        association = dist_df[ids]
        neg_df = pd.DataFrame({'Gene_A': gene1_neg, 'Gene_B': gene2_neg, 'Association Strength': association})

    else:
        r_ids, c_ids = np.where(dist_df > th_up)
        gene1_pos = gene_ids1[r_ids]
        gene2_pos = gene_ids2[c_ids]
        association = dist_df[r_ids, c_ids]
        pos_df = pd.DataFrame({'Gene_A': gene1_pos, 'Gene_B': gene2_pos, 'Association Strength': association})

        r_ids, c_ids = np.where(dist_df < th_down)
        gene1_neg = gene_ids1[r_ids, c_ids]
        gene2_neg = gene_ids2[r_ids, c_ids]
        association = dist_df.values[r_ids, c_ids]

        neg_df = pd.DataFrame({'Gene_A': gene1_neg, 'Gene_B': gene2_neg, 'Association Strength': association})

    return pos_df, neg_df


def batchPairConstruction(df1, df2=None, n_batches=1, method='pearson', pos_criterion='5%', neg_criterion='5%', **kwds):
    # TODO: parallelize
    if df2 is None:
        n_cols = df1.shape[1]
        break_points = np.linspace(0, n_cols, num=n_batches+1, dtype=np.int)

        pos_pairs, neg_pairs = [], []

        for i in range(len(break_points) - 1):
            for j in range(i + 1):
                pos_, neg_ = calculateAssociationSelf(df1.iloc[:, break_points[i]:break_points[i + 1]],
                                                 method=method, as_pairs=True, pos_criterion=pos_criterion,
                                                 neg_criterion=neg_criterion, **kwds)
                print(neg_)
                print(pos_)
                pos_pairs += [pos_]
                neg_pairs += [neg_]
    else:
        n_cols1, n_cols2 = df1.shape[1], df2.shape[1]
        break_points1 = np.linspace(0, n_cols1, num=n_batches+1, dtype=np.int)
        break_points2 = np.linspace(0, n_cols2, num=n_batches+1, dtype=np.int)

        pos_pairs, neg_pairs = [], []

        for i in range(len(break_points1) - 1):
            for j in range(len(break_points2) - 1):
                pos_, neg_ = calculateAssociationOther(df1.iloc[:, break_points1[i]:break_points1[i + 1]],
                                     df2.iloc[:, break_points2[j]:break_points2[j + 1]],
                                     method=method, as_pairs=True, pos_criterion=pos_criterion,
                                     neg_criterion=neg_criterion, **kwds)
                pos_pairs += [pos_]
                neg_pairs += [neg_]

    pos_pairs = pd.concat(pos_pairs, axis=0, ignore_index=True)
    neg_pairs = pd.concat(neg_pairs, axis=0, ignore_index=True)

    return pos_pairs, neg_pairs


def findAberrantDataPoints1D(data, nsigma_merge=2, nsigma_aberrant=3, aberrant_side='both',
                             plot_aberrant=False, partition_method='KDE', **kwargs):
    '''
    Function to define aberrant data points in 1D data
    :param data: the 1D data
    :param nsigma_bound: sigma bound, between which data is considered abnormal
    :param side:
    :param kwargs:
    :return:
    '''
    olabels = np.empty(data.shape)
    olabels[:] = np.nan
    nanmask = ~np.isnan(data)
    data = data[nanmask]

    if partition_method == 'KDE':
        labels, kde = getPartitionsKDE(data, return_model=True,  **kwargs)
    elif partition_method == 'KMEANS':
        km = KMeans(n_clusters=5, random_state=0).fit(data[..., None])
        labels = km.labels_

    partitionMeans, partitionSTD, partitionCounts = [], [], []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        partitionMeans += [np.mean(data[labels == label])]
        partitionSTD += [np.std(data[labels == label])]
        partitionCounts += [np.sum(labels == label)]

    partitionMeans, partitionCounts, partitionSTD = np.array(partitionMeans), np.array(partitionCounts),\
                                                    np.array(partitionSTD)

    largest_partition = np.argmax(partitionCounts)
    merge_mask = (partitionMeans > (partitionMeans[largest_partition] - nsigma_merge * partitionSTD[largest_partition])) & \
                 (partitionMeans < (partitionMeans[largest_partition] + nsigma_merge * partitionSTD[largest_partition]))

    merge_labels = labels[merge_mask]
    normal_mask = np.array([l in merge_labels for l in labels])

    normal_mean, normal_std = np.mean(data[normal_mask]), np.std(data[normal_mask])

    if aberrant_side.lower() == 'lower':
        aberrant_partitions = partitionMeans < (normal_mean - nsigma_aberrant * normal_std)
    elif aberrant_side.lower() == 'higher':
        aberrant_partitions = partitionMeans > (normal_mean + nsigma_aberrant * normal_std)
    else:
        aberrant_partitions = (partitionMeans > (normal_mean + nsigma_aberrant * normal_std)) | \
                              (partitionMeans < (normal_mean - nsigma_aberrant * normal_std))

    aberrant_points = 1 * aberrant_partitions

    if plot_aberrant:
        plotlabels = np.array(['Normal' if point == 0 else 'Aberrant' for point in aberrant_points])
        plotLabelledHistogram(data, plotlabels)

    olabels[nanmask] = aberrant_points
    return olabels


def findVAFSubpopulations(data, min_clonal_var_fraction=0.05, clon_vaf=0.4, subclonal_vaf=0.3,
                             plot_aberrant=False, savename_plot=None, partition_method='KDE', **kwargs):

    '''
    Script to determine automatically if a VAF partition is clonal or not
    :param data: VAF data, assumed to be 1D
    :param min_clonal_var_fraction: NOT YET IN USE
    :param clon_vaf: the minimum average VAF a population needs to have in order to be considered clonal
    :param subclonal_vaf: NOT YET IN USE
    :param plot_aberrant: whether the resulting populations need to be plotted
    :param partition_method: KDE or Kmeans
    :param kwargs: arguments to be passed on to the getPartitionsKDE function
    :return:
    '''

    olabels = np.empty(data.shape)
    olabels[:] = np.nan
    nanmask = ~np.isnan(data)
    data = data[nanmask]

    if partition_method == 'KDE':
        labels, kde = getPartitionsKDE(data, return_model=True,  **kwargs)
    elif partition_method == 'KMEANS':
        km = KMeans(n_clusters=5, random_state=0).fit(data[..., None])
        labels = km.labels_

    partitionMeans, partitionCounts = [], []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        partitionMeans += [np.mean(data[labels == label])]
        partitionCounts += [np.sum(labels == label)]

    partitionMeans, partitionCounts = np.array(partitionMeans), np.array(partitionCounts)

    label_mapper = {vaf: 'Superclonal' for vaf in unique_labels[partitionMeans > 0.5]}

    mask = partitionMeans < 0.55

    partitionMeans, partitionCounts, unique_labels = partitionMeans[mask], partitionCounts[mask], \
                                                     unique_labels[mask]
    sort_ids = partitionMeans.argsort()[::-1]
    partitionMeans, partitionCounts, unique_labels = partitionMeans[sort_ids], partitionCounts[sort_ids], \
                                                     unique_labels[sort_ids]
    #total_vars = np.sum(partitionCounts)
    id_clonal, id_subclonal, id_unknown = 1, 1, 1

    first_sub = True
    for id, mean in enumerate(partitionMeans):

        if mean >= clon_vaf:
            tag = 'Clonal ' + str(id_clonal)
            id_clonal += 1

        elif (mean < clon_vaf) & (mean >= subclonal_vaf):
            tag = 'Unknown ' + str(id_unknown)
            id_unknown += 1

        else:
            tag = 'Subclonal ' + str(id_subclonal)
            id_subclonal += 1


        label_mapper[unique_labels[id]] = tag

    olabels = np.array([label_mapper[i] for i in labels])

    if plot_aberrant:
        print('Saving figure ...')
        plotLabelledHistogram(data, olabels, savepath=savename_plot)

    return olabels


def findCCFSubpopulations(data, plot_aberrant=False, savename_plot=None, partition_method='KDE', **kwargs):
    '''
    Script to determine automatically if a VAF partition is clonal or not
    :param data: VAF data, assumed to be 1D
    :param min_clonal_var_fraction: NOT YET IN USE
    :param clon_vaf: the minimum average VAF a population needs to have in order to be considered clonal
    :param subclonal_vaf: NOT YET IN USE
    :param plot_aberrant: whether the resulting populations need to be plotted
    :param partition_method: KDE or Kmeans
    :param kwargs: arguments to be passed on to the getPartitionsKDE function
    :return:
    '''

    olabels = np.empty(data.shape)
    olabels[:] = np.nan
    nanmask = ~np.isnan(data)
    data = data[nanmask]

    if partition_method == 'KDE':
        labels, kde = getPartitionsKDE(data, return_model=True,  **kwargs)

    elif partition_method == 'KMEANS':
        km = KMeans(n_clusters=5, random_state=0).fit(data[..., None])
        labels = km.labels_

    partitionMeans, partitionCounts = [], []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        partitionMeans += [np.mean(data[labels == label])]
        partitionCounts += [np.sum(labels == label)]

    partitionMeans, partitionCounts = np.array(partitionMeans), np.array(partitionCounts)

    label_mapper = {}

    sort_ids = partitionMeans.argsort()[::-1]
    partitionMeans, partitionCounts, unique_labels = partitionMeans[sort_ids], partitionCounts[sort_ids], \
                                                     unique_labels[sort_ids]

    id_clonal, id_subclonal = 1, 1

    for id, mean in enumerate(partitionMeans):

        if mean >= 0.5:
            tag = 'Clonal ' + str(id_clonal)
            id_clonal += 1

        else:
            tag = 'Subclonal ' + str(id_subclonal)
            id_subclonal += 1

        label_mapper[unique_labels[id]] = tag

    olabels = np.array([label_mapper[i] for i in labels])

    if plot_aberrant:
        print('Saving figure ...')
        plotLabelledHistogram(data, olabels, savepath=savename_plot, xlabel='CCF')

    return olabels


def plotLabelledHistogram(data, labels=None, num_bins=100, xlabel='', savepath=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    bin_seq = np.linspace(np.min(data), np.max(data), num=num_bins)

    if labels is None:
        labels = ['']

    unique_plotlabels = np.unique(labels)

    for i, label_ in enumerate(unique_plotlabels):
        ax.hist(data[labels == label_], bins=bin_seq, label=label_, density=False)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.grid(False)
    #plt.legend(fontsize=16)
    ax.get_legend().remove()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if savepath is not None:
        print('Saving figure to: ' + savepath)
        plt.savefig(savepath, dpi=1000, bbox_inches="tight")

    #plt.show()


class PartitionedKDE(KernelDensity):

    def __init__(self, bandwidth=1.0, algorithm='auto',
                 kernel='gaussian', metric="euclidean", atol=0, rtol=0,
                 breadth_first=True, leaf_size=40, metric_params=None,
                 height_threshold=None):

        self.height_threshold = height_threshold

        super().__init__(bandwidth=bandwidth, algorithm=algorithm,
                 kernel=kernel, metric=metric, atol=atol, rtol=rtol,
                 breadth_first=breadth_first, leaf_size=leaf_size, metric_params=metric_params)

        self.n_partitions = None

    def get_minima(self, xmin, xmax, step_size=0.001):

        # Make sure code continues if an array of only 0's is passed
        if xmin == xmax:
            xmax = xmin + 1
        xs = np.arange(xmin, xmax, step_size)
        ys = np.exp(self.score_samples(xs[:, None]))

        if self.height_threshold is not None:
            min_vals = []
            min_i = argrelextrema(ys, np.less)[0]
            max_i = argrelextrema(ys, np.greater)[0]
            # find maxima surrounding each minimum
            for i in min_i:
                pos = bisect_left(max_i, i)
                own_height = ys[i]
                if pos == 0:
                    left_height = np.inf
                else:
                    left_height = ys[max_i[pos - 1]]
                if pos == len(max_i):
                    right_height = np.inf
                else:
                    right_height = ys[max_i[pos]]
                # see if the distance to both maxima is above the threshold
                if min(left_height, right_height) - own_height >= self.height_threshold:
                    min_vals.append(xs[i])
        else:
            min_vals = xs[argrelextrema(ys, np.less)[0]]

        return min_vals

    def partition_data(self, data):
        self.fit(data[:, None])
        min_vals = self.get_minima(np.min(data), np.max(data))

        labels = np.zeros(data.shape, dtype=np.int)
        labels = pd.Series(labels, index=data.index)

        for min_ in min_vals:
            labels += 1 * (data < min_)

        self.n_partitions = len(min_vals) + 1

        return labels

    def bic(self, data):
        data = data[:, None]
        return np.log(data.shape[0]) * self.n_partitions - self.score(data)#_samples(data)

    def crossValidateBandwidth(self, data, range=(0.01, 0.1)):
        best_param = 0
        score = np.inf
        for bw in np.linspace(range[0], range[1], 10):
            splits = ShuffleSplit(n_splits=10, random_state=0, test_size=0.1, train_size=None)
            scores = []
            for train_index, test_index in splits.split(data):
                self.bandwidth = bw
                self.partition_data(data[train_index])
                scores.append(self.bic(data[test_index]))
            mean_score = np.mean(scores)
            if mean_score < score:
                score = mean_score
                best_param = bw
        self.bandwidth = best_param

    def selectBandwidth(self, data, range=(0.01, 0.1)):
        best_param = 0
        score = np.inf
        for bw in np.linspace(range[0], range[1], 10):
            self.bandwidth = bw
            self.partition_data(data)
            new_score = self.bic(data)
            if new_score < score:
                score = new_score
                best_param = bw
        self.bandwidth = best_param

    def plotHistogramKDE(self, data, xlabel='', savepath=None, ax=None,
                         annotated_patients=None, annotation_labels=None):

        min_vals = self.get_minima(np.min(data), np.max(data))
        xs = np.arange(np.min(data), np.max(data), 0.001)
        ys = np.exp(self.score_samples(xs[:, None]))

        labels = np.zeros(xs[:].shape)

        for min_ in min_vals:
            labels += 1 * (xs < min_)

        labels = np.array(['Cluster ' + str(np.int(label) + 1) for label in labels])

        mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                                                        'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                                                        'tab:olive', 'tab:cyan'])

        num_bins = 200
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        bin_seq = np.linspace(np.min(data), np.max(data), num=num_bins)

        if labels is None:
            labels = ['']

        unique_plotlabels = np.unique(labels)
        weights = np.ones_like(data)/(len(data) * np.diff(bin_seq)[0])
        ax.hist(data, bins=bin_seq, label='Data', weights=weights)

        for i, label_ in enumerate(unique_plotlabels):
            p = ax.plot(xs[labels == label_], ys[labels == label_], label=label_)
            color = p[0].get_color()
            ax.fill_between(xs[labels == label_], 0, ys[labels == label_], alpha=0.7, color=color)

        if annotated_patients is not None:
            ax.hist(data[annotated_patients], bins=bin_seq,
                    label=annotation_labels, weights=weights[:len(annotated_patients)])

        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel('Density', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.grid(False)
        plt.legend(fontsize=16)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if savepath is not None:
            print('Saving figure to: ' + savepath)
            plt.savefig(savepath, dpi=1000, bbox_inches="tight")
            plt.close()

        # plt.show()


def SetParamsAndPartition(gene, data, KDE):
    print(gene)
    d = data.df.loc[:, gene]
    KDE.selectBandwidth(d)
    return KDE.partition_data(d)

def applyKDEBinarization(data, height_threshold=None, remove_zv=True, parallel=False, n_jobs=4):
    all_labels = []
    KDE = PartitionedKDE(height_threshold=height_threshold)
    if parallel:
        all_labels = Parallel(n_jobs=n_jobs)(delayed(SetParamsAndPartition)(g, data=data, KDE=KDE) for g in data.genes())
    else:
        all_labels = [SetParamsAndPartition(g, data, KDE) for g in data.genes()]
    df = pd.DataFrame(all_labels, index=data.genes()).transpose()
    return DiscreteOmicsDataSet(df, type=data.type, patient_axis=0, remove_zv=remove_zv)
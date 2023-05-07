import pandas as pd
import numpy as np
from NetworkAnalysis.ContinuousOmicsDataSet import ContinuousOmicsDataSet
from NetworkAnalysis.InteractionNetwork import UndirectedInteractionNetwork

n_pat, n_genes = 10, 100
rand_mat = 8 * np.random.rand(n_pat, n_genes)

testdataset_expr = ContinuousOmicsDataSet(pd.DataFrame(rand_mat,
                                                       index=['P' + str(i) for i in range(n_pat)],
                                                       columns=['Gene' + str(i) for i in range(n_genes)]),
                                       remove_zv=False, remove_nas=False,
                                       average_dup_genes=False,
                                       average_dup_samples=False, patient_axis='auto')


pos, neg = testdataset_expr.getPosNegPairs(method='kendall', pos_criterion=1e-2)

X_train, X_test, Y_train, Y_test, summary = testdataset_expr.getTrainTestPairs(corr_measure='ranked-pearson', return_summary=True)

graph_ = UndirectedInteractionNetwork(pos, colnames=('Gene_A', 'Gene_B'))
degrees = graph_.getDegreeDF(return_names=False)

degrees.index = degrees.Gene.values

adj_dict = self.getAdjDict(return_names=False)

counts = degrees.sort_values(by='Gene', ascending=False)['Count'].values * neg_pos_ratio
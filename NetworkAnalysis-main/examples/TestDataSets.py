from NetworkAnalysis.ContinuousOmicsDataSet import ContinuousOmicsDataSet, densityDiscretization1D
from NetworkAnalysis.DiscreteOmicsDataSet import DiscreteOmicsDataSet
from NetworkAnalysis.OmicsDataSet import OmicsDataSet
import pandas as pd
import numpy as np
import time

DATA_PATH = '/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/data_metabric/'

# Test continuous dataset
genexp = pd.read_csv(DATA_PATH + 'data_expression.txt', sep='\t',  header=0)
genexp = genexp.set_index('Hugo_Symbol')

genexp = genexp.drop(['Entrez_Gene_Id'], axis=1)
genexp = genexp.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

cont_expr = ContinuousOmicsDataSet(genexp)


cont_expr.plotExpressionRegime('GATA3', max_regimes=5, annotation_labels=None, remove_frame=True, method='LALALA')

# cont_expr.compareGeneProfiles(['ESR1', 'TP53', 'FOXA1'])
# cont_expr.compareSampleProfiles()

start = time.time()
test_bin = cont_expr.applyGMMBinarization(max_regimes=5)
stop = time.time()

print('Old method took %f seconds.' % (stop-start))

start = time.time()
test_bin2 = cont_expr.applyGMMBinarization_new(max_regimes=5)
stop = time.time()

print('New method took %f seconds.' % (stop-start))


#print(np.unique(test_bin.df.values, return_counts=True))


# Test the discrete omics class
test_df = pd.read_csv(DATA_PATH + 'data_expression_binarized.txt', sep='\t', index_col=0, header=0)
test_df = test_df[test_df.columns.values[(test_df.sum() > 600) & (test_df.sum() < (test_df.shape[0] - 600))]]

expr_up = DiscreteOmicsDataSet(test_df, attrs={1: ' +', 0: ' -'})
expr_down = DiscreteOmicsDataSet(1 - test_df)

test = expr_up.getSignificantGenePairs(expr_up, count_thresh=20, pvals_thresh=np.log(1e-30))

'''
# test the scaling algorithms
test_df = ContinuousOmicsDataSet(pd.DataFrame(np.array([[5,4,3],[2,1,4],[3,4,6],[4,2,8]]), index=['A', 'B', 'C', 'D']), patient_axis=0)
print(test_df.df)


test_df.normalizeProfiles()
print(test_df.df)

# in general the idea of using 150 count as threshold is sound

# test the new code
N = 10000
N = 10
random_mat = np.random.randint(5, size=(N, N)).astype(np.int)

#method 1: np
import time

start = time.time()
P = np.matmul(random_mat, random_mat)
U = np.triu(P, 1)
end = time.time()

print('Execution of np.matmul last %d seconds' % (end - start) )

# method 2

start = time.time()
R, C = np.triu_indices(N, 1)
P2 = np.einsum('ij,ij->j', random_mat[:, R], random_mat[:, C]) # Raises a memory error
end = time.time()
'''''

import numpy as np


# Test the density discretization
test = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
test = genexp.transpose()['ERBB2'].values

labels = densityDiscretization1D(test)
print(labels)
print(np.sum(labels == 1))


n, bins, patches = plt.hist(np.sort(test), color='#ff7f0e', bins=100)
max_val = np.max(np.sort(test)[labels==0])
for i in np.arange(len(patches))[bins[1:] > max_val]:

    patches[i].set_fc('#1f77b4')

plt.show()


# step 1: calculate distances
v = test
v = np.sort(v)
v_diff = v[1:] - v[:-1]

min_clustersize = 3
# step 2: select the areas with the lowest densities
nclusters = 1
min_ = np.min(v_diff)
v_diff[:min_clustersize] = min_
v_diff[-min_clustersize:] = min_
print(v_diff)

import matplotlib.pyplot as plt
from scipy.stats import expon

fig, ax = plt.subplots(1, 1)

ax.plot(np.sort(v_diff), expon.pdf(np.sort(v_diff), scale=np.mean(v_diff)), 'r-', lw=5, alpha=0.6, label='expon pdf')

plt.hist(v_diff, bins=100, alpha=0.4, histtype='stepfilled')
ax.legend(loc='best', frameon=False)
plt.show()

expon.sf(np.max(v_diff), scale=np.mean(v_diff))

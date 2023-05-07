from NetworkAnalysis.DiscreteOmicsDataSet import DiscreteOmicsDataSet
import pandas as pd
from NetworkAnalysis.ContinuousOmicsDataSet import ContinuousOmicsDataSet
import numpy as np

# The DiscreteOmicsDataset is a pandas object, with the samples as rows and the genes as columns.

# the Dataset object can be initialized like a dataframe, or using a dataframe:
df = pd.DataFrame(np.array([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]),
                                                    index=['P1', 'P2', 'P3'],
                                                    columns=['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene3'])

testdataset_mut = DiscreteOmicsDataSet(data = np.array([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]),
                                                    index=['P1', 'P2', 'P3'],
                                                    columns=['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene3'],
                                       type='MUT', remove_zv=True, remove_nas=True,
                                       average_dup_genes=True,
                                       average_dup_samples=False, patient_axis='auto')

testdataset_mut = DiscreteOmicsDataSet(data = np.array([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]),
                                                    index=['P1', 'P2', 'P3'],
                                                    columns=['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene3'],
                                       type='MUT', remove_zv=True, remove_nas=True,
                                       average_dup_genes=True,
                                       average_dup_samples=False, patient_axis='auto')



# The different options during intialization help the user to filter the dataframe

# To access the underlying dataframe:
print(testdataset_mut)

# As such all pandas commands can be applied directly on the df attribute
sum_over_genes = testdataset_mut.sum(axis=1)

# If one want to have the sample/gene IDs of the data:
sample_ids = testdataset_mut.samples(as_set=False)
gene_ids = testdataset_mut.genes(as_set=False)

# To obtain a deep copy of an object:
deep_copy_mut = testdataset_mut.deepcopy()

# To only only select a subset of the genes we can
gene_subset = testdataset_mut.subsetGenes(['Gene1', 'Gene2'], inplace=False)
sample_subset = testdataset_mut.subsetSamples(['P1', 'P2'], inplace=False)

# To map sample/gene ids onto their new ids, we can use the following function:
complete_map = {'Gene1': 1, 'Gene2': 2, 'Gene3': 3, 'Gene4': 4, 'Gene5': 5}
deep_copy_mut.mapGeneIDs(complete_map)
print(deep_copy_mut.genes())

# If we have an incomplete map, we can do:
deep_copy_mut = testdataset_mut.deepcopy()

incomplete_map = {'Gene1': 1, 'Gene2': 2, 'Gene3': 3}
deep_copy_mut.mapGeneIDs(incomplete_map, method='lossy')
print(deep_copy_mut.genes())

deep_copy_mut = testdataset_mut.deepcopy()
deep_copy_mut.mapGeneIDs(incomplete_map, method='conservative')
print(deep_copy_mut.genes())

# In the same way we can map the samples
complete_map = {'P1': 1,  'P2': 2, 'P3': 3}
deep_copy_mut.mapSampleIDs(complete_map)
print(deep_copy_mut.samples())

# If we have an incomplete map, we can do:
deep_copy_mut = testdataset_mut.deepcopy()

incomplete_map = {'P1': 1,  'P2': 2}
deep_copy_mut.mapSampleIDs(incomplete_map, method='lossy')
print(deep_copy_mut.samples())

deep_copy_mut = testdataset_mut.deepcopy()
deep_copy_mut.mapSampleIDs(incomplete_map, method='conservative')
print(deep_copy_mut.samples())

# To quickly visualize the discrete gene profiles:
testdataset_mut.compareGeneProfiles()

# To visualize the sample profiles:
testdataset_mut.compareSampleProfiles(sort=False)

# Note that the subset command can be used in combination with the plotting functions,
# to visualize a subset of the data
testdataset_mut.subsetGenes(['Gene1', 'Gene2']).compareGeneProfiles()

# In mutation data, we often remove the low-frequent genes. For instance,
# removing all genes with a mutation rate below 40%:
testdataset_mut_hf = testdataset_mut.thresholdGenes(count_thresh=0.4, inplace=False)

# Now suppose that we have a second Dataset:
testdataset_expr = DiscreteOmicsDataSet(pd.DataFrame(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                                     index=['P1', 'P2', 'P4'],
                                                     columns=['Gene1', 'Gene2', 'Gene3']), type='CNA')

# Then we can easily filter to keep the common samples/genes:
mut_common_genes, expr_common_genes = testdataset_mut.keepCommonGenes([testdataset_expr])
mut_common_samples, expr_common_samples = testdataset_mut.keepCommonGenes([testdataset_expr])
mut_common_genes.getCommonSamples(expr_common_genes)

import pandas as pd
from NetworkAnalysis.OmicsDataSet import OmicsDataSet

testdic = {'col 1': [11, 21], 'col 2': [12, 22], 'col 3': [13, 23]}
pd.DataFrame(testdic).__repr__()
testdataset = OmicsDataSet(testdic, index=['Sample 1', 'Sample 2'])

testdataset.genes()

testdataset.subsetGenes(['col 1', 'col 2'])

# OmicsDataSet supports different types of indexing:
testdataset[[1, 0], :]
testdataset[['col 1', 'col 2']]
testdataset[:, ['col 1', 'col 2']]
testdataset[['Sample 1'],  ['col 1', 'col 2']]
testdataset[:, [True, False]]

# OmicsData even supports fancy mixed indexing, with bools strings and integers
testdataset[[0, 1],  ['col 1', 'col 2']]

testdic2 = {'col 2': [11, 21], 'col 4': [14, 24], 'col 5': [15, 25]}
testdataset2 = OmicsDataSet(testdic2, index=['Sample 1', 'Sample 2'])

testdataset.getCommonGenes(testdataset2)
testdataset.getCommonSamples(testdataset2)

subset_test1, subset_test2 = testdataset.keepCommonGenes(testdataset2)

genemap = {'col 1': 'Col 1', 'col 2': 'Col 1', 'col 3': 'Col 3'}
testdataset.mapGeneIDs(genemap, aggr_duplicates='mean')

samplemap = {'Sample 1': 'Patient 1', 'Sample 2': 'Patient 1'}
testdataset.mapSampleIDs(samplemap, aggr_duplicates='mean')

zv_test = {'col 1': [11, 11], 'col 2': [12, 12], 'col 3': [13, 23]}
zv_testdata = OmicsDataSet(zv_test).removeZeroVarianceGenes()

testdataset2.getAllPairs()
testdataset2.logtransform()
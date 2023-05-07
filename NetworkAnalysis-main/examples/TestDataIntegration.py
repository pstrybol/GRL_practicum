import pandas as pd
import numpy as np
from DiscreteOmicsDataSet import DiscreteOmicsDataSet
from IntegrationAnalysis import IncludeAttributes
from InteractionNetwork import UndirectedInteractionNetwork

DATA_PATH = '/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/data_metabric/'

# load some data from the METABRIC study to test the integration
cna_data = pd.read_csv(DATA_PATH + 'cna_data_preprocessed.csv', header=0, index_col=0)
mut_data = pd.read_csv(DATA_PATH + 'mut_data_preprocessed.csv', header=0, index_col=0)
diff_genexp = pd.read_csv(DATA_PATH + 'data_expression_binarized.txt', sep='\t', index_col=0, header=0)

# check if the code deals with NAs appropriately, i.e. by removing the gene
cna_data = 1*(cna_data > 0) + 2*(cna_data < 0)
cna_dataset = DiscreteOmicsDataSet(cna_data, attrs={2.: ' del', 0.: ' ', 1.: ' amp'}, remove_nas=True)

mut_dataset = DiscreteOmicsDataSet(mut_data, attrs={0.: ' ', 1.: ' mut'}, patient_axis=0)
gx_dataset  = DiscreteOmicsDataSet(diff_genexp, attrs={0.: ' -', 1.: ' +'})

gx_dataset.keepCommonPatients([mut_dataset, cna_dataset])

# Define a known query
query = DiscreteOmicsDataSet(np.minimum(1-gx_dataset.df['ESR1'], mut_dataset.df['TP53']),
                             attrs={1: 'condition', 0: 'no condition'}, patient_axis=0, remove_nas=False)

# read in the network
network = UndirectedInteractionNetwork.from_file(DATA_PATH + 'BIOGRID-ORGANISM-Homo_sapiens-3.4.161.tab2.txt', sep='\t',
                                       header=0, col_names=['Official Symbol Interactor A', 'Official Symbol Interactor B'])

# Perform the integration
cyto_df = IncludeAttributes([gx_dataset, mut_dataset, cna_dataset], query,
                             network, testtype='right', count_thresh=20,
                             pvals_threshs=[np.log(1e-45), np.log(1e-20), np.log(1e-20)])

cyto_df2 = cyto_df.loc[np.array(['+' in s for s in cyto_df.Target_Attr]) |
                       np.array(['+' in s for s in cyto_df.Source_Attr]) |
                       np.array(['-' in s for s in cyto_df.Target_Attr]) |
                       np.array(['-' in s for s in cyto_df.Source_Attr])]


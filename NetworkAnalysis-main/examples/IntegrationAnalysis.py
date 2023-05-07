import numpy as np
from scipy.stats import binom
import pandas as pd
from DiscreteOmicsDataSet import DiscreteOmicsDataSet
from InteractionNetwork import UndirectedInteractionNetwork

def queryDatasets(discrete_datasets, query, testtype='right', count_thresh=20, pvals_threshs=np.log(1e-20)):
    try:
        _ = (e for e in pvals_threshs)
    except TypeError:
        pvals_thresh = [pvals_threshs]

    if len(pvals_threshs) != len(discrete_datasets):

        if len(pvals_threshs) == 1:
            pvals_threshs = [pvals_threshs[0] for i in range(len(discrete_datasets))]

        else:
            raise ValueError('Please specify the p-value threshold using an scalar or '
                             'an  array that is equal to the number of datasets presented.')

    count_id = 0
    pval_df = None

    for dataset in discrete_datasets:

        if count_id == 0:
            pval_df = dataset.getSignificantGenePairs(query, count_thresh=count_thresh, pvals_thresh=pvals_threshs[count_id],
                                                      testtype=testtype, check_ones_only=True)
            print('Dataset 1, % d pairs identified.' % (pval_df.shape[0]))

        else:
            pval_ = dataset.getSignificantGenePairs(query, count_thresh=count_thresh, pvals_thresh=pvals_threshs[count_id],
                                                    testtype=testtype,  check_ones_only=True)
            pval_df = pd.concat([pval_df, pval_], ignore_index=True, axis=0)
            print('Dataset %d, %d pairs identified.' % (count_id+1, pval_.shape[0]))
            print(pval_)

        count_id += 1

    return pval_df


def mapPairsOnNetwork(pval_df, network, gx_dataset=None, mut_data=None, prob_thresh = 0.5, attrs=None):

    genenames = [s.split(' ')[0] for s in pval_df.Gene_A]

    interaction_df = network.checkInteraction_list(genenames).interactions
    print('%f interactions identified.' % interaction_df.shape[0])
    print(interaction_df)

    if gx_dataset is not None:
        N_pats = len(gx_dataset.samples)
        N_orig_pairs = interaction_df.shape[0]
        counter_a, counter_b = 0., 0.
        gene_as, gene_bs, gene_a_attrs, gene_b_attrs = [], [], [], []
        gx_attrs, mut_attrs = gx_dataset.attrs, mut_data.attrs

        for row_id in range(interaction_df.shape[0]):
            genomic_aberr_a, genomic_aberr_b = False, False

            gene_a = interaction_df.Gene_A.iloc[row_id]
            gene_b = interaction_df.Gene_B.iloc[row_id]

            attr_a, attr_b = tuple(attrs[gene_a]), tuple(attrs[gene_b])

            if gx_attrs[0][1:] in attr_a:
                gx_gene_a = 1-gx_dataset.df[gene_a]
            elif gx_attrs[1][1:] in attr_a:
                gx_gene_a = gx_dataset.df[gene_a]
            else:  # This means that the expression of the gene is unrelated to the query, take the most occurring value
                genomic_aberr_a = True
                counter_a += 1.

            if gx_attrs[0][1:] in attr_b:
                gx_gene_b = 1-gx_dataset.df[gene_b]
            elif gx_attrs[1][1:] in attr_b:
                gx_gene_b = gx_dataset.df[gene_b]
            else:
                genomic_aberr_b = True
                counter_b += 1.

            if (not genomic_aberr_a) and (not genomic_aberr_b):
                P_a_given_b = gx_gene_a.dot(gx_gene_b)/gx_gene_b.sum()
                P_b_given_a = gx_gene_a.dot(gx_gene_b)/gx_gene_a.sum()

                if (P_a_given_b >= prob_thresh) or (P_b_given_a >= prob_thresh):

                    if P_b_given_a >= P_a_given_b:
                        gene_as.append(gene_a)
                        gene_bs.append(gene_b)

                        gene_a_attrs.append(attr_a)
                        gene_b_attrs.append(attr_b)

                    else:
                        gene_as.append(gene_a)
                        gene_bs.append(gene_b)

                        gene_a_attrs.append(attr_b)
                        gene_b_attrs.append(attr_a)
            #TODO: figure out what to do with this, using  intrinsic types of datasets
            elif genomic_aberr_a:
                pass
            else:
                pass

            if mut_attrs[1][1:] in gene_a_attrs:
                pass

        interaction_df = pd.DataFrame({'Gene_A': gene_as, 'Gene_B': gene_bs})
    print('Fraction of pairs that have unrelated expression %d' % (1.*counter_a/N_orig_pairs))
    print('Fraction of pairs that have unrelated expression %d' % (1.*counter_b/N_orig_pairs))

    return interaction_df


def IncludeAttributes(discrete_datasets, query, network, testtype='right', count_thresh=20, pvals_threshs=np.log(1e-20)):
    # problem interaction df only contains the

    pval_df = queryDatasets(discrete_datasets, query, testtype=testtype,
                            count_thresh=count_thresh, pvals_threshs=pvals_threshs)
    # print(pval_df.sort_values(by='p-value'))

    # carefully handle the attributes to avoid double mapping
    gene_names = np.array([s.split(' ')[0] for s in pval_df.Gene_A])
    unique_genes = set(gene_names)
    gene_attrs = np.array([s.split(' ')[1] if len(s.split(' ')) == 2 else ' ' for s in pval_df.Gene_A])

    # remove
    gene_names = gene_names[gene_attrs != ' ']
    gene_attrs = gene_attrs[gene_attrs != ' ']
    attr_genes = set(gene_names)

    attrs = {gene: tuple(set(gene_attrs[gene_names == gene])) if gene in attr_genes else ' ' for gene in unique_genes}

    for query_name in query.genes:
        attrs[query_name] = query_name

    interaction_df = mapPairsOnNetwork(pval_df, network, discrete_datasets[0], attrs=attrs)
    # make sure that the expression dataset is the first one in the list !!!!!!!!!!!!!!!

    interaction_df['Source_Attr'] = list(map(lambda x: tuple(attrs[x]), interaction_df.Gene_A))
    interaction_df['Target_Attr'] = list(map(lambda x: tuple(attrs[x]), interaction_df.Gene_B))

    return interaction_df
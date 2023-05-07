import pandas as pd
import numpy as np
from scipy.stats import binom
from OmicsAnalysis.DiscreteOmicsDataSet import get_pval_one_mat
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict

''' The aim of CNSegmenter is to group genes that have a similar amplification/deletion profile, 
    prior to any downstream analysis '''

# TODO: write tests!!!!!!


class CNSegmenter:

    def __init__(self, gene_names, gene_starts, gene_ends, gene_chroms, gene2attribute=None):

        self.gene_names = check_array(gene_names, "gene_names", str)
        self.gene_starts = check_array(gene_starts, "gene_starts", np.int)
        self.gene_ends = check_array(gene_ends, "gene_ends", np.int)
        self.gene_chroms = check_array(gene_chroms, "gene_chroms", str)
        self.segments = None
        self.gene2attribute = gene2attribute

        assert len(self.gene_chroms) == len(self.gene_ends) == len(self.gene_starts) == len(self.gene_names), \
            "All arrays should have the same length and be in the same order."

        assert len(np.unique(gene_names)) == len(gene_names), "All genes should have a unique identifier"

        uniq_contigs = np.unique(self.gene_chroms)
        print("Initialized %i genes for %i contigs." % (len(self.gene_names), len(uniq_contigs)))


    @classmethod
    def from_DF(cls, df, gene_name_col="Gene", start_col="START", stop_col="STOP", chrom_col="CHROM"):
        df = df.dropna()
        return cls(gene_names=df[gene_name_col].values,
                   gene_starts=df[start_col].values,
                   gene_ends=df[stop_col].values,
                   gene_chroms=df[chrom_col].values)

    @classmethod
    def from_gaf(cls,
                 filepath,
                 feature_col="FeatureType",
                 feature_type='gene',
                 id_col='FeatureID',
                 gene_col="Gene",
                 pos_col="CompositeCoordinates"):
        '''
        Based on the specification given in https://www.ensembl.org/info/website/upload/gff.html
        :param filepath:
        :param gene_col:
        :param start_col:
        :param stop_col:
        :param chrom_col:
        :param keep: {'first', 'last', False}, default 'first', keep only first feature_id in case of duplicates
        - ``first`` : Mark duplicates as ``True`` except for the
          first occurrence.
        - ``last`` : Mark duplicates as ``True`` except for the
          last occurrence.
        - False : Mark all duplicates as ``True``.
        :return:
        '''

        gaf_file = pd.read_csv(filepath, sep='\t', header=0, engine='python',
                               usecols=[feature_col, id_col, pos_col, gene_col])
        gaf_file = gaf_file.loc[gaf_file[feature_col].apply(lambda s: str(s).lower()) == feature_type]

        feature_ids = gaf_file[id_col].apply(lambda s: str(s).split("|")[0])
        mask = ~feature_ids.duplicated().values
        feature_ids = feature_ids.values[mask]
        gaf_file = gaf_file.loc[mask]

        chroms = gaf_file[pos_col].apply(lambda s: str(s).split(":")[0]).values
        starts = gaf_file[pos_col].apply(lambda s: str(s).split(":")[1].split("-")[0]).values
        stops = gaf_file[pos_col].apply(lambda s: str(s).split(":")[1].split("-")[-1]).values
        genes = gaf_file[gene_col].apply(lambda s: str(s).split("|")[0]).values

        gene2attribute = dict(zip(genes, feature_ids))

        return cls(gene_names=feature_ids,
                   gene_starts=starts,
                   gene_ends=stops,
                   gene_chroms=chroms,
                   gene2attribute=gene2attribute)

    @classmethod
    def from_gtf(cls,
                 filepath,
                 gene_col=8,
                 start_col=3,
                 stop_col=4,
                 chrom_col=0,
                 feature_col=2,
                 feature_type='gene',
                 attribute='gene_name'):
        '''
        Based on the specification given in https://www.ensembl.org/info/website/upload/gff.html
        :param filepath:
        :param gene_col:
        :param start_col:
        :param stop_col:
        :param chrom_col:
        :return:
        '''

        trans_dict = {ord(c): None for c in ' "\''}
        gtf_file = pd.read_csv(filepath, sep='\t', header=None, engine='python', comment='#')
        gtf_file = gtf_file.loc[gtf_file.iloc[:, feature_col].apply(lambda s: str(s).lower()) == feature_type]

        attributes = gtf_file.iloc[:, gene_col].apply(lambda s: str(s).split(attribute)[-1].split(';')[0]
                                                    .translate(trans_dict)).values.flatten()

        gene_ids = gtf_file.iloc[:, gene_col].apply(lambda s: str(s).split("gene_id")[-1].split(';')[0]
                                                    .translate(trans_dict)).values.flatten()

        gene2attribute = defaultdict(list)

        for gene_id, attribute_id in zip(gene_ids, attributes):
            gene2attribute[gene_id].append(attribute_id)

        for i, (gene_id, attribute_id) in enumerate(gene2attribute.items()):
            if attribute_id == ["nan"]:
                attributes[i] = gene_id
                gene2attribute[gene_id] = gene_id

        gtf_file['parsed_attribute'] = attributes

        other_cols = list(gtf_file.columns.values[[start_col, stop_col, chrom_col]]) + ['parsed_attribute']
        gtf_file = gtf_file[other_cols]

        gtf_file = gtf_file.dropna()
        gtf_file = gtf_file.loc[gtf_file.parsed_attribute != 'nan']
        gtf_file = gtf_file.drop_duplicates()

        genes, counts = np.unique(gtf_file.iloc[:, 3].values, return_counts=True)
        uniq_genes = genes[counts < 2]
        gtf_file = gtf_file.loc[gtf_file.iloc[:, 3].isin(uniq_genes)]

        return cls(gene_names=gtf_file.iloc[:, 3].values,
                   gene_starts=gtf_file.iloc[:, 0].values,
                   gene_ends=gtf_file.iloc[:, 1].values,
                   gene_chroms=gtf_file.iloc[:, 2].values, gene2attribute=gene2attribute)

    @classmethod
    def from_tsv(cls,
                 filepath,
                 gene_col="gene_name",
                 start_col="start",
                 stop_col="end",
                 chrom_col="seqname"):

        '''
        Defaults are chosen to import from TCGA reference tsv
        '''
        usecols = [gene_col, start_col, stop_col, chrom_col]
        df = pd.read_csv(filepath, usecols=usecols, sep='\t')
        df = df.drop_duplicates()
        gene_names, counts = np.unique(df.gene_name.values, return_counts=True)

        uniq_genes = gene_names[counts == 1]
        df = df.loc[df[gene_col].isin(uniq_genes)]

        return cls(gene_names=df[gene_col].values,
                    gene_starts=df[start_col].values,
                    gene_ends=df[stop_col].values,
                    gene_chroms=df[chrom_col].values)

    def check_segmentation_status(self):
        if self.segments is None:
            raise IOError("Please initialize the segments first,"
                          " using the <group_coocurring_genes> function.")

    def check_genes_on_same_chrom(self, gene_list):
        mask = np.isin(self.gene_names, gene_list)

        chroms = np.unique(self.gene_chroms[mask])

        if len(chroms) > 1:
            raise IOError("The genes are not all from the same chromosome.")


    @property
    def gene2segment(self):
        self.check_segmentation_status()
        return dict([(g, k) for k, v in self.segments.items() for g in v])

    @property
    def unique_chroms(self):
        return np.unique(self.gene_chroms)

    def subset_chrom(self, chrom):
        mask = self.gene_chroms == chrom

        return CNSegmenter(self.gene_names[mask],
                           self.gene_starts[mask],
                           self.gene_ends[mask],
                           self.gene_chroms[mask])

    def __repr__(self):
        return self.to_DF().head().__repr__()

    def __str__(self):
        return self.to_DF().head().__repr__()

    def __getitem__(self, gene_name):

        if self.segments is None:
            raise IOError("Please initialize the segments first, using the <group_coocurring_genes> function.")

        if gene_name not in self.gene_names:
            warnings.warn("%s is not a known gene name." % gene_name)
            return None

        for v in self.segments.values():
            if gene_name in v:
                return v

    def copy(self):
        return CNSegmenter(gene_names=self.gene_names,
                           gene_starts=self.gene_starts,
                           gene_ends=self.gene_ends,
                           gene_chroms=self.gene_chroms)

    def sort_genes_by_position(self, gene_list, sort_by='middle'):
        '''

        :param gene_list:
        :param sort_by:
        :return:
        '''
        mask = np.isin(self.gene_names, gene_list)

        chroms = np.unique(self.gene_chroms[mask])

        if len(chroms) > 1:
            raise IOError("The genes are not all from the same chromosome.")

        if sort_by.lower() == 'start':
            idx = np.argsort(self.gene_starts[mask])

        elif sort_by.lower() == 'end':
            idx = np.argsort(self.gene_ends[mask])

        else:
            middles = (self.gene_starts[mask] + self.gene_ends[mask])/2.
            idx = np.argsort(middles)

        return self.gene_names[mask][idx]

    def group_coocurring_genes(self, cnv_data, pval_thresh=1e-5, patience=0):

        uniq_chroms = np.unique(self.gene_chroms)
        gene_dict = {}
        common_genes = np.intersect1d(cnv_data.columns.values, self.gene_names)

        if len(common_genes) == 0:
            raise IOError("No genes in common between the provided data and the annotations."
                          "Check that both are provided in the same format.")
        else:
            self.subset_genes(genes_to_keep=common_genes)

        for chr in uniq_chroms:
            mask = self.gene_chroms == chr

            chrom_dict = search_segments_recursively(self.gene_starts[mask],
                                                     self.gene_ends[mask],
                                                     self.gene_names[mask],
                                                     cnv_data,
                                                     pval_thresh=pval_thresh,
                                                     patience=patience)

            gene_dict = {**gene_dict, **chrom_dict}

        self.segments = gene_dict
        return gene_dict

    def subset_genes(self, genes_to_keep, inplace=True):
        mask = np.isin(self.gene_names, genes_to_keep)

        if inplace:
            self.gene_chroms = self.gene_chroms[mask]
            self.gene_ends = self.gene_ends[mask]
            self.gene_names = self.gene_names[mask]
            self.gene_starts = self.gene_starts[mask]

        else:
            return CNSegmenter(self.gene_names[mask], self.gene_starts[mask],
                               self.gene_ends[mask], self.gene_chroms[mask])

    def to_DF(self):
        return pd.DataFrame({"Gene": self.gene_names,
                             "Start": self.gene_starts,
                             "Stop": self.gene_ends,
                             "Chrom": self.gene_chroms}
                            )

    def get_chrom(self, gene):
        return self.gene_chroms[self.gene_names == gene]

    def get_genes_in_chrom(self, chrom):
        return self.gene_names[self.gene_chroms == chrom]

    def get_chrom_length(self, chrom):
        mask = self.gene_chroms == chrom

        return get_total_segment_length(self.gene_starts[mask], self.gene_ends[mask])

    def get_total_length(self):
        uniq_chroms = np.unique(self.gene_chroms)
        total_length = 0

        for chrom in uniq_chroms:
            total_length += self.get_chrom_length(chrom)

        return total_length

    def get_gene_length(self):
        """
        Advanced function to return gene_lengths.
        Works best if performed using exons, by providing the gene2attribute  field
        :return:
        """
        if self.gene2attribute is None:
            warnings.warn("<gene2attribute> is not provided. The length per attribute will not account for possible overlaps.")
            return pd.Series((self.gene_ends - self.gene_starts).sum(), index=self.gene_names)

        else:
            length_dict = {}
            for gene, attributes in self.gene2attribute.items():

                mask = np.isin(self.gene_names, attributes)
                length_dict[gene] = get_total_segment_length(self.gene_starts[mask], self.gene_ends[mask])

            return pd.Series(length_dict)

    def get_segment_length(self, gene_list):
        mask = np.isin(self.gene_names, gene_list)

        chroms = np.unique(self.gene_chroms[mask])

        if len(chroms) > 1:
            raise IOError("The genes are not all from the same chromosome.")

        start = np.min(self.gene_starts[mask])
        stop = np.max(self.gene_ends[mask])

        return stop - start

    def return_neighbors(self, gene, n_neighbors=5):
        mask = self.gene_chroms == self.get_chrom(gene)
        centers = (self.gene_ends[mask] - self.gene_starts[mask])/2.

        idx = np.argsort(centers)
        genes = self.gene_names[mask][idx]

        goi_id = np.where(genes == gene)[0][0]

        upper_id = np.minimum(len(genes), goi_id + n_neighbors + 1).astype(int)
        lower_id = np.maximum(0, goi_id - n_neighbors).astype(int)

        return genes[lower_id:upper_id]

    def return_overlapping_genes(self, gene):
        mask = self.gene_chroms == self.get_chrom(gene)
        starts, ends, gene_names = self.gene_starts[mask], self.gene_ends[mask], self.gene_names[mask]

        goi_id = np.where(gene_names == gene)[0][0]

        start_g, end_g = starts[goi_id], ends[goi_id]

        mask = ((starts <= start_g) & (start_g <= ends)) | \
               ((starts <= end_g) & (end_g <= ends)) | \
               ((starts >= start_g) & (end_g >= ends))

        return gene_names[mask]

    def plot_segment(self, cnv_data, genes=None, n_neighbors=5, sort_genes=False,
                     return_df=False,
                     show_overlapping_genes=False, **plot_kwargs):
        '''
        Plots the events in a copy number matrix that are assumed to be binary
        :param cnv_data: binary data containing copy number events.
        Typically these are either amplifications or deletions.
        :param genes: The genes that need to be plotted.
        Can either be a iterable of genes, or a single gene (string).
        In case of the latter, the n_neighbors parameter specifies
        how many genes around the provided gene are shown.
        :param n_neighbors: only used if genes is a string.
        The number of neighboring genes plotted around the provided gene.
        :param show_overlapping_genes: only used if genes is a string.
        Whether to show all genes that overlap with the query genes.
        :return:
        '''

        if isinstance(genes, str):
            if show_overlapping_genes:
                genes = self.return_overlapping_genes(genes)
            else:
                genes = self.return_neighbors(genes, n_neighbors=n_neighbors)

        genes_in_data = cnv_data.columns.values
        common_genes = [gene for gene in genes if gene in genes_in_data]

        plot_df = cnv_data[common_genes]

        return plot_cohort(plot_df, sort_genes=sort_genes, return_df=return_df, **plot_kwargs)

    def cluster_by_chrom(self, score_df, cnv_data, pval_thresh=1e-5, keep_scored_genes_only=False):
        assert isinstance(score_df, pd.Series), "Provided scores must be a Series object."
        genes_with_score = score_df.index.values

        common_genes = np.intersect1d(genes_with_score, self.gene_names)

        if len(common_genes) == 0:
            raise IOError("No genes in common between provided scores and annotations.")

        else:
            score_df = score_df.loc[score_df.index.isin(common_genes)]

        if keep_scored_genes_only:
            cns_ = self.subset_genes(common_genes, inplace=False)

        else:
            cns_ = self.copy()

        uniq_chroms = cns_.unique_chroms
        cluster_dict = {}

        for chrom in uniq_chroms:
            genes = cns_.get_genes_in_chrom(chrom)
            score_df_chrom = score_df.loc[score_df.index.isin(genes)]

            clusters_chrom = get_clusters_ranked_list(cnv_data, score_df_chrom, pval_thresh=pval_thresh)

            cluster_dict = {**clusters_chrom, **cluster_dict}

        return cluster_dict

    def get_all_genes_between(self, gene_list, return_statistics=True):
        '''
        Returns all genes between a segment, between a list of genes.
        If the genes provided are not all from the same segment, the code throws an error.
        This way, this function can also be used to check this requirement.
        :param gene_list: genes to check
        :return: a numpy array containing all the genes in the interval.
        '''

        mask = np.isin(self.gene_names, gene_list)

        chroms = np.unique(self.gene_chroms[mask])

        if len(chroms) > 1:
            raise IOError("The genes are not all from the same chromosome.")

        middles = (self.gene_starts[mask] + self.gene_ends[mask])/2.

        max_idx, min_idx = np.argmax(middles), np.argmin(middles)
        max_middle, min_middle = middles[max_idx], middles[min_idx]
        max_gene, min_gene = self.gene_names[mask][max_idx], self.gene_names[mask][min_idx]

        chrom_mask = self.gene_chroms == chroms[0]
        middles = (self.gene_starts[chrom_mask] + self.gene_ends[chrom_mask])/2.
        mask = (min_middle <= middles) & (middles <= max_middle)

        return_genes = self.gene_names[chrom_mask][mask]

        if return_statistics:
            missing_genes = np.setdiff1d(return_genes, gene_list)
            print('############################################################################')
            print("There were %i genes provided."%len(gene_list))
            print('The outermost genes were: %s <-----------> %s' % (min_gene, max_gene))
            print('%i genes from that interval were not present in the provided list.' % len(missing_genes))
            print('############################################################################')

        return return_genes

    def template_per_chromosome(self, cnv_df):

        common_genes = np.intersect1d(cnv_df.columns.values, self.gene_names)
        cnv_df = cnv_df.copy(deep=True)[common_genes]

        cns_ = self.subset_genes(common_genes, inplace=False)
        uniq_chroms = cns_.unique_chroms
        odf = []
        clusterdict = {}

        for chrom in uniq_chroms:
            genes = cns_.get_genes_in_chrom(chrom)
            df = template_cnv_df(cnv_df[genes])
            odf.append(df.copy(deep=True))

            row_sums_templates = np.sum(df.values, axis=0, keepdims=False)[..., None]
            df[df == 0] = -1

            truth = df.transpose().dot(cnv_df[genes]) == row_sums_templates
            r, c = np.where(truth)

            r_genes, c_genes = df.columns.values, cnv_df.columns.values
            clusters = {r_genes[ri]: genes[c[r == ri]] for ri in r}

            clusterdict = {**clusterdict, **clusters}

        odf = pd.concat(odf, axis=1)
        return odf, clusterdict


def template_cnv_df(cnv_df):
    '''

    :param cnv_df:
    :return:cnv_filtered
    '''

    cnv_filtered = cnv_df.transpose().drop_duplicates()
    return cnv_filtered.transpose()


def check_array(arr, arr_name, arr_type):

    if arr is None:
        return None

    try:
        arr = np.asarray(arr).astype(arr_type)

    except ValueError:
        raise IOError("Make sure that %s can be converted to %s." % (arr_name, arr_type.__name__))

    if arr.dtype == np.floating:
        assert not np.any(np.isnan(arr)), "Make sure that %s does not contain nans." % arr_name

    return arr


def search_segments(starts, stops, names, cnv_data, pval_thresh=1e-5, patience=0):
    '''
    Obtains the segments of genes that are strongly co-occuring in their amplification/deletions across patients.
    Each segment is represented by the gene that has the most events as this can be considered the "focal" gene.
    :param starts: start coordinates of the genes.
    :param stops: stop coordinates of the genes.
    :param names: ids of the genes.
    :param cnv_data: copy number data, expected binary, either denoting deletions or amplifications.
    :param pval_thresh: the threshold on the p-value for the co-occurence test.
    :return: a dict containing the focal gene as key and all genes belonging to that segment as values.
    '''
    middles = (starts + stops)/2.  # taking the middle of the gene is safer in case there are overlapping genes
    sort_idx = np.argsort(middles)
    names = names[sort_idx]

    cnv_data = cnv_data[names]
    most_affected_genes = cnv_data[names].sum(axis=0).sort_values(ascending=True).index.to_list()
    cnv_data = cnv_data.values

    region_dict = {}

    while len(most_affected_genes) > 0:
        curr_gene = most_affected_genes.pop()

        all_genes_in_segment = search_for_gene(curr_gene, names, cnv_data, pval_thresh, patience)
        most_affected_genes = [g for g in most_affected_genes if g not in all_genes_in_segment]

        region_dict[curr_gene] = all_genes_in_segment

    return region_dict


def search_segments_recursively(starts, stops, names, cnv_data, pval_thresh=1e-5, patience=0,
                                region_dict=None):
    '''
    Obtains the segments of genes that are strongly co-occuring in their amplification/deletions across patients.
    Each segment is represented by the gene that has the most events as this can be considered the "focal" gene.
    :param starts: start coordinates of the genes.
    :param stops: stop coordinates of the genes.
    :param names: ids of the genes.
    :param cnv_data: copy number data, expected binary, either denoting deletions or amplifications.
    :param pval_thresh: the threshold on the p-value for the co-occurence test.
    :return: a dict containing the focal gene as key and all genes belonging to that segment as values.
    '''
    middles = (starts + stops)/2.  # taking the middle of the gene is safer in case there are overlapping genes
    sort_idx = np.argsort(middles)
    names, starts, stops = names[sort_idx], starts[sort_idx], stops[sort_idx]
    names2int = {g: i for i, g in enumerate(names)}

    cnv_data = cnv_data[names]
    most_affected_genes = cnv_data[names].sum(axis=0).sort_values(ascending=True).index.to_list()
    cnv_mat = cnv_data.values

    if region_dict is None:
        region_dict = {}

    while len(most_affected_genes) > 0:
        curr_gene = most_affected_genes.pop()

        all_genes_in_segment = search_for_gene(curr_gene, names, cnv_mat, pval_thresh, patience)
        idx = [names2int[g] for g in all_genes_in_segment]
        region_dict[curr_gene] = all_genes_in_segment

        min_idx, max_idx = min(idx), max(idx)
        if min_idx > 0:
            search_segments_recursively(starts[:min_idx], stops[:min_idx], names[:min_idx],
                                        cnv_data, pval_thresh=pval_thresh, patience=patience,
                                        region_dict=region_dict)

        if max_idx < (len(starts) - 1):
            search_segments_recursively(starts[(max_idx + 1):],
                                        stops[(max_idx + 1):],
                                        names[(max_idx + 1):],
                                        cnv_data, pval_thresh=pval_thresh, patience=patience,
                                        region_dict=region_dict)

        sel_genes = set([g for l in region_dict.values() for g in l])
        most_affected_genes = [g for g in most_affected_genes if g not in sel_genes]

    return region_dict


def search_for_gene(query_gene, genes, cnv_data, pval_thresh=1e-5, patience=0):
    '''
    Performs the search given a query (focal) to find all genes that co-occur
    :param query_gene: str
    :param genes: np.array of strings
    :param cnv_data:
    :param pval_thresh:
    :return: a list containing all genes in that segment
    '''
    query_id = np.where(genes == query_gene)[0][0]
    curr_id = query_id

    genes_in_segment = [query_gene]
    counter = 0
    temp_genes = []

    while (counter <= patience) and curr_id < (len(genes) - 1):
        curr_id += 1

        pval = get_pval_cooc(cnv_data[:, query_id], cnv_data[:, curr_id])

        if pval <= pval_thresh:
            temp_genes.append(genes[curr_id])
            genes_in_segment += temp_genes
            counter = 0
            temp_genes = []

        else:
            temp_genes.append(genes[curr_id])
            counter += 1

    curr_id = query_id
    counter = 0
    temp_genes = []

    while (counter <= patience) and curr_id > 0:
        curr_id -= 1

        pval = get_pval_cooc(cnv_data[:, query_id], cnv_data[:, curr_id])

        if pval <= pval_thresh:
            temp_genes.append(genes[curr_id])
            genes_in_segment += temp_genes
            counter = 0
            temp_genes = []

        else:
            temp_genes.append(genes[curr_id])
            counter += 1

    return genes_in_segment


def get_pval_cooc(v1, v2):
    '''
    Co-occurence test currently used, relying on the binomial distribution
    :param v1:
    :param v2:
    :return:
    '''
    n_samples = len(v1)
    p1, p2 = v1.sum()/n_samples, v2.sum()/n_samples
    cooc = np.dot(v1, v2)

    pval = binom.sf(cooc, n_samples, p1 * p2)

    return pval


def get_pval_cooc_mat(mat, v):
    '''
    Co-occurence test currently used, relying on the binomial distribution
    :param mat:
    :param v:
    :return:
    '''
    n_samples = len(v)
    p1, p2 = v.sum()/n_samples, mat.sum(axis=1)/n_samples
    cooc = np.matmul(mat, v)

    pval = binom.sf(cooc, n_samples, p1 * p2)

    return pval


def sort_by_rows(df, sort_genes=True):
    nrows, ncols = df.shape
    cols, index = df.columns.values, df.index.values
    plot_df = df.copy()

    if sort_genes:
        plot_df = plot_df.append(df.sum(axis=0), ignore_index=True)
        plot_df = plot_df.sort_values(by=nrows, axis=1, ascending=False)
        plot_df = plot_df.drop(nrows, axis=0)

    plot_df['Rowsum'] = np.matmul(plot_df.values, 2. ** np.arange(0, -ncols, -1))
    plot_df.index = index

    plot_df = plot_df.sort_values(by='Rowsum', ascending=False)
    plot_df = plot_df.drop('Rowsum', axis=1)

    return plot_df


def plot_cohort(df, save_path=None, sort=True, sort_genes=True, ax=None,
                return_df=False, cmap='Greys', fontsize_ylabel=14, fontsize_xlabels=14,
                remove_grid=True, sample_spacing=None,
                **plot_kwargs):

    if sort:
        plot_df = sort_by_rows(df, sort_genes=sort_genes)

    else:
        plot_df = df.copy(deep=True)

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))

    ax.pcolor(plot_df, cmap=cmap, **plot_kwargs)

    ax.set_xticks(np.arange(0.5, len(plot_df.columns), 1))
    ax.set_xticklabels(plot_df.columns, fontsize=fontsize_xlabels, rotation=45)

    ax.set_yticklabels([])
    ax.set_yticks([])

    if remove_grid:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    if sample_spacing is not None:
        y_min, y_max = ax.get_ylim()
        for i in np.arange(1, len(plot_df.columns), 1):
            ax.vlines(x=i, color='w', ymin=y_min, ymax=y_max, linewidth=sample_spacing)

    ax.set_ylabel('Samples' + ' (n=' + str(df.shape[0]) + ')', fontsize=fontsize_ylabel)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        if ax is None:
            plt.show()

    if return_df:
        return plot_df
    else:
        return 0


def get_overlapping_dict(starts, ends, gene_names):
    coords = np.hstack((starts, ends))
    start_flag = np.array([True] * len(starts) + [False] * len(ends))
    gene_names = np.hstack((gene_names, gene_names))

    idx = np.argsort(coords)
    gene_names = gene_names[idx]
    start_flag = start_flag[idx]
    curr_genes = []
    overlap_dict = {}

    for flag, gene in zip(start_flag, gene_names):

        if flag:
            overlap_dict[gene] = [g for g in curr_genes]

            for g in curr_genes:
                overlap_dict[g].append(gene)

            curr_genes.append(gene)

        else:
            curr_genes.remove(gene)

    return overlap_dict


def get_clusters_ranked_list(cnv_data, score_df, pval_thresh=1e-5):
    '''
    Cluster genes from a ranked list.
    Clustering is done starting from the highest gene in the list.
    All genes that are significantly co-occuring with this gene are added to the cluster of this
    and no longer considered.
    :param cnv_data:
    :param score_df:
    :param pval_thresh:
    :return:
    '''

    assert isinstance(cnv_data, pd.DataFrame), "input data must be a pandas DF."
    common_genes = np.intersect1d(score_df.index.values, cnv_data.columns.values)

    score_df = score_df.copy(deep=True).loc[score_df.index.isin(common_genes)]
    cnv_data = cnv_data.copy(deep=True)[common_genes]

    genes_in_order = score_df.loc[common_genes].sort_values(ascending=False).index.values
    odict = {}

    while len(genes_in_order) > 1:
        best_gene = genes_in_order[0]
        genes_in_order = np.delete(genes_in_order, 0)

        pvals = get_pval_cooc_mat(cnv_data[genes_in_order].values.transpose(), cnv_data[best_gene].values)

        cluster_genes = genes_in_order[pvals < pval_thresh]
        odict[best_gene] = np.insert(cluster_genes, 0, best_gene)

        genes_in_order = genes_in_order[~np.isin(genes_in_order, cluster_genes)]

    if len(genes_in_order) == 1:
        odict[genes_in_order[0]] = genes_in_order

    return odict


def post_process_segments(segment_dict, cnv_data, pval_thresh):
    # check if some segments are not co-ocurring to each other, this step is optional.
    segments = list(segment_dict.keys())
    cooc_segments = get_pval_one_mat(cnv_data[segments], pvals_thresh=pval_thresh, count_thresh=1)

    # TODO: finish this
    # filter the cooc segments


def get_total_segment_length(start1, end1):
    '''
    Get the total length of N possibly overlapping segments, indicated by a start and end position.

    :param start1: starting points of the segments
    :param end1: end points of the segments
    :return: a vector of the same length as varpos, containing the CN status for that vector.
    '''

    start1, end1 = np.array(start1, dtype=np.int), np.array(end1, dtype=np.int)
    status = np.ones(len(start1))

    status_ids = status, -1 * status

    T = np.concatenate([start1, end1])
    S = np.concatenate(status_ids)

    sort_ids = np.argsort(T)
    S = S[sort_ids]
    T = T[sort_ids]

    cumS = np.cumsum(S)

    # find zeros
    zero_ids = np.where(cumS < 1e-5)[0][:-1]
    total_length = T[-1]
    empty_intervals = sum(T[idx + 1] - T[idx] for idx in zero_ids)

    return total_length - empty_intervals


def test_get_total_segment_length():
    test_starts = np.array([1, 3, 7])
    test_ends = np.array([4, 5, 10])

    total_length = get_total_segment_length(test_starts, test_ends)

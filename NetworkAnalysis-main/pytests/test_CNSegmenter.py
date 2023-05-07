import numpy as np
import pandas as pd
import sys
# move the current working directory to the end position
sys.path = sys.path[1:] + sys.path[:1]
from OmicsAnalysis.CNSegmenter import CNSegmenter, search_for_gene, search_segments,\
    search_segments_recursively, get_overlapping_dict, get_pval_cooc_mat, get_pval_cooc,\
    get_clusters_ranked_list, template_cnv_df


def test_CN_init():
    cns = CNSegmenter(gene_names=np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6']),
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=np.array(["chr1"] * 6))

    cns = CNSegmenter(gene_names=np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6']),
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)


def test_to_DF():
    cns = CNSegmenter(gene_names=np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6']),
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)
    df = cns.to_DF()
    assert df.shape[1] == 4


def test_subset_genes():
    cns = CNSegmenter(gene_names=np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6']),
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    cns.subset_genes(["G1", "G2"])

    assert 2 == len(cns.gene_names)

    cns.subset_genes(["G1", "g2"])

    assert 1 == len(cns.gene_names)


def test_search_segments():
    segment_starts = np.array([0, 5, 10, 15, 20, 25])
    segment_ends = np.array([4, 9, 14, 19, 24, 29])
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cnv_data = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    odict = search_segments(segment_starts, segment_ends, gene_names, cnv_df, pval_thresh=0.06)

    assert len(odict) == 3


def test_search_segments_recursively():
    segment_starts = np.array([0, 5, 10, 15, 20, 25])
    segment_ends = np.array([4, 9, 14, 19, 24, 29])
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cnv_data = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    odict = search_segments_recursively(segment_starts, segment_ends, gene_names, cnv_df, pval_thresh=0.06)

    print(odict)
    correct_segments = [{'G1', 'G2', 'G3'}, {'G4'}, {'G5', 'G6'}]
    segments = [set(l) for l in odict.values()]

    assert all(seg in correct_segments for seg in segments)

    segment_starts = np.array([0, 5, 10, 15, 20, 25, 28, 35, 40])
    segment_ends = np.array([4, 9, 14, 19, 24, 29, 34, 69, 55])
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9'])

    cnv_data = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1],
                         [1, 1, 1, 0, 0, 0, 1, 1, 1],
                         [1, 1, 1, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0, 0]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    odict = search_segments_recursively(segment_starts, segment_ends, gene_names, cnv_df, pval_thresh=0.06)

    print(odict)
    segments = [set(l) for l in odict.values()]

    correct_segments = [{'G1', 'G2', 'G3'}, {'G4'}, {'G5', 'G6'}, {'G7', 'G8', 'G9'}]

    assert all(seg in correct_segments for seg in segments)

    segment_starts = np.array([0, 5, 10, 15, 20, 25, 28, 35, 0])
    segment_ends = np.array([4, 9, 14, 19, 24, 29, 34, 69, 20])
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9'])

    cnv_data = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1],
                         [1, 1, 1, 0, 0, 0, 1, 1, 1],
                         [1, 1, 1, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0, 0]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    odict = search_segments_recursively(segment_starts, segment_ends, gene_names, cnv_df, pval_thresh=0.06)

    print(odict)
    segments = [set(l) for l in odict.values()]

    correct_segments = [{'G1', 'G2', 'G3', 'G9'}, {'G4'}, {'G5', 'G6'}, {'G7', 'G8'}]

    assert all(seg in correct_segments for seg in segments)


def test_from_tsv():
    filepath = "/home/mlarmuse/Downloads/gencode.gene.info.v22 (1).tsv"
    segm_obj = CNSegmenter.from_tsv(filepath)

    assert len(segm_obj.gene_names) > 0


def test_search_for_gene():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cnv_data = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    odict1 = search_for_gene('G1', gene_names, cnv_data, pval_thresh=0.06)
    odict3 = search_for_gene('G3', gene_names, cnv_data, pval_thresh=0.06, patience=0)

    odict4 = search_for_gene('G4', gene_names, cnv_data, pval_thresh=0.06)
    odict5 = search_for_gene('G5', gene_names, cnv_data, pval_thresh=0.06)

    print(odict1)
    print(odict3)
    print(odict4)
    print(odict5)

    odict1 = search_for_gene('G1', gene_names, cnv_data, pval_thresh=0.06, patience=1)
    odict3 = search_for_gene('G3', gene_names, cnv_data, pval_thresh=0.06, patience=1)

    print(odict1)
    print(odict3)


def test_group_coocurring_genes():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    cnv_data = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    odict = cns.group_coocurring_genes(cnv_df)

    print(odict)

    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 0, 6, 15]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 6, 12, 21]),
                      gene_chroms=["chr1"] * 6 + ["chr2"] * 3)

    cnv_data = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 1],
                         [1, 1, 1, 0, 0, 0, 1, 1, 0],
                         [1, 1, 1, 0, 0, 0, 1, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 1, 0, 0, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    odict2 = cns.group_coocurring_genes(cnv_df)

    print(odict2)


def test_getitem():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    correct_behaviour = False
    try:
        cns["hihi"]

    except IOError:
        correct_behaviour = True

    assert correct_behaviour

    cnv_data = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    odict = cns.group_coocurring_genes(cnv_df, pval_thresh=0.06)

    assert set(cns["G1"]) == set(cns["G3"])
    assert cns["G19"] is None


def test_gene2segment():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    cnv_data = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    odict = cns.group_coocurring_genes(cnv_df, pval_thresh=0.06)

    mapper = cns.gene2segment
    mapped_genes = [mapper[g] for g in cns.gene_names]

    assert set(mapped_genes) == set(list(odict.keys()))


def test_get_chrom():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    assert all(cns.get_chrom(g) == "chr1" for g in gene_names)
    assert len(cns.get_chrom("nonsense")) == 0


def test_get_genes_in_chrom():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    genes_in_chr1 = cns.get_genes_in_chrom("chr1")

    assert {'G1', 'G2', 'G3', 'G4', 'G5', 'G6'} == set(genes_in_chr1)
    assert len(cns.get_genes_in_chrom("chr2")) == 0

    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 0, 6, 15]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 6, 12, 21]),
                      gene_chroms=["chr1"] * 6 + ["chr2"] * 3)

    genes_in_chr1 = cns.get_genes_in_chrom("chr1")

    assert {'G1', 'G2', 'G3', 'G4', 'G5', 'G6'} == set(genes_in_chr1)
    assert len(cns.get_genes_in_chrom("chr2")) == 3


def test_return_neighbors():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    nbs = cns.return_neighbors("G3", n_neighbors=1)

    assert {'G2', 'G3', 'G4'} == set(nbs)

    nbs = cns.return_neighbors("G3", n_neighbors=2)
    assert {'G1', 'G2', 'G3', 'G4', 'G5'} == set(nbs)

    nbs = cns.return_neighbors("G3", n_neighbors=3)
    assert np.all(gene_names == nbs)

    nbs = cns.return_neighbors("G3", n_neighbors=30)
    assert np.all(gene_names == nbs)


def test_plot_segment():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    cnv_data = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    exit_code = cns.plot_segment(cnv_df, ['G1', 'G2', 'G3'])
    assert exit_code == 0

    exit_code = cns.plot_segment(cnv_df, 'G2')
    assert exit_code == 0

    exit_code = cns.plot_segment(cnv_df, 'G2', n_neighbors=3, sort_genes=True)
    assert exit_code == 0

    exit_code = cns.plot_segment(cnv_df, 'G2', n_neighbors=3, sort_genes=False)
    assert exit_code == 0

    exit_code = cns.plot_segment(cnv_df, 'G2', n_neighbors=3, sort_genes=True, sample_spacing=10)
    assert exit_code == 0

    exit_code = cns.plot_segment(cnv_df, 'G2', n_neighbors=3, sort_genes=True, sample_spacing=10,
                                 remove_grid=False)
    assert exit_code == 0

    exit_code = cns.plot_segment(cnv_df, 'G2', n_neighbors=3, sort_genes=True, sample_spacing=10,
                                 remove_grid=False, fontsize_xlabels=25)
    assert exit_code == 0

    df = cns.plot_segment(cnv_df, 'G2', n_neighbors=3, sort_genes=True, sample_spacing=10,
                                 remove_grid=False, fontsize_xlabels=25, return_df=True)

    assert df.shape[0] == 6


def test_get_overlap_dict():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    odict = get_overlapping_dict(cns.gene_starts, cns.gene_ends, cns.gene_names)

    assert all(not l for l in odict.values())

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 12]),
                      gene_ends=np.array([6, 11, 16, 21, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    odict = get_overlapping_dict(cns.gene_starts, cns.gene_ends, cns.gene_names)

    assert {'G3', 'G4', 'G5'} == set(odict['G6'])


def test_return_overlapping_genes():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 12]),
                      gene_ends=np.array([6, 11, 16, 21, 24, 29]),
                      gene_chroms=["chr1"] * 6)

    overlap_genes = cns.return_overlapping_genes("G4")
    assert {'G3', 'G5', 'G6'} == set(list(overlap_genes))


def test_get_pval_mat_cooc():

    cnv_data = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1]])

    v = np.array([1, 1, 1, 0, 0])

    pval_vect = get_pval_cooc_mat(cnv_data.transpose(), v)
    pval_v = [get_pval_cooc(cnv_data[:, i], v) for i in range(cnv_data.shape[1])]

    assert np.all(np.abs(np.array(pval_v) - pval_vect) < 1e-5)


def test_get_clusters_ranked_list():

    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 30]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 35]),
                      gene_chroms=["chr1"] * 7)

    cnv_data = np.array([[1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 0]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)
    score_df = pd.Series([5, 4, 4, 3, 3, 1, 1], index=gene_names)

    odict = get_clusters_ranked_list(cnv_df, score_df, pval_thresh=0.1)
    print(odict)

    # improve testing, insert asserts


def test_cluster_by_chrom():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 0, 6, 15]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 6, 12, 21]),
                      gene_chroms=["chr1"] * 6 + ["chr2"] * 3)

    cnv_data = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    scores = pd.Series(np.arange(9), index=gene_names)

    odict = cns.cluster_by_chrom(scores, cnv_df, pval_thresh=1e-1)

    print(odict)


def test_get_all_genes_between():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 0, 6, 15]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 6, 12, 21]),
                      gene_chroms=["chr1"] * 6 + ["chr2"] * 3)

    all_genes = cns.get_all_genes_between(['G1', 'G5'])

    assert {'G1', 'G2', 'G3', 'G4', 'G5'} == set(list(all_genes))

    correct_behavior = False

    try:
        all_genes = cns.get_all_genes_between(['G1', 'G8'])

    except IOError:
        correct_behavior = True

    assert correct_behavior

    all_genes = cns.get_all_genes_between(['G9', 'G7'])

    assert {'G7', 'G8', 'G9'} == set(list(all_genes))


def test_copy():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 0, 6, 15]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 6, 12, 21]),
                      gene_chroms=["chr1"] * 6 + ["chr2"] * 3)

    cns2 = cns.copy()

    cns.subset_genes(['G1', 'G3', 'G5'], inplace=True)

    assert len(cns2.gene_names) == 9
    assert len(cns.gene_names) == 3


def test_template_cnv_df():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cnv_data = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    filtered_df = template_cnv_df(cnv_df)

    print(filtered_df)


def test_template_per_chromosome():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 0, 6, 15]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 6, 12, 21]),
                      gene_chroms=["chr1"] * 6 + ["chr2"] * 3)

    cnv_data = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    filtered_df, cluster_dict = cns.template_per_chromosome(cnv_df)

    assert {'G1', 'G4', 'G5', 'G7'} == set(filtered_df.columns.to_list())
    assert {'G1', 'G2', 'G3'} == set(list(cluster_dict['G1']))
    assert {'G4'} == set(list(cluster_dict['G4']))
    assert {'G5', 'G6'} == set(list(cluster_dict['G5']))
    assert {"G7", "G8", "G9"} == set(list(cluster_dict['G7']))

    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 0, 6, 15]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 6, 12, 21]),
                      gene_chroms=["chr1"] * 6 + ["chr2"] * 3)

    cnv_data = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1]])

    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8"])
    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    filtered_df, cluster_dict = cns.template_per_chromosome(cnv_df)

    assert {'G1', 'G4', 'G5', 'G7'} == set(filtered_df.columns.to_list())
    assert {'G1', 'G2', 'G3'} == set(list(cluster_dict['G1']))
    assert {'G4'} == set(list(cluster_dict['G4']))
    assert {'G5', 'G6'} == set(list(cluster_dict['G5']))
    assert {"G7", "G8"} == set(list(cluster_dict['G7']))

    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 0, 6, 15]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 6, 12, 21]),
                      gene_chroms=["chr1"] * 9)

    cnv_data = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1]])

    cnv_df = pd.DataFrame(cnv_data, columns=gene_names)

    filtered_df, cluster_dict = cns.template_per_chromosome(cnv_df)

    assert {'G1', 'G4', 'G5', 'G7'} == set(filtered_df.columns.to_list())
    assert {'G1', 'G2', 'G3'} == set(list(cluster_dict['G1']))
    assert {'G4'} == set(list(cluster_dict['G4']))
    assert {'G5', 'G6'} == set(list(cluster_dict['G5']))
    assert {"G7", "G8", "G9"} == set(list(cluster_dict['G7']))

    assert np.all(np.unique(filtered_df.values) == np.array([0, 1]))


def test_get_chrom_length():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 0, 6, 15]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 6, 12, 21]),
                      gene_chroms=["chr1"] * 6 + ["chr2"] * 3)

    chr1_len = cns.get_chrom_length('chr1')
    chr2_len = cns.get_chrom_length('chr2')

    assert 29 == chr1_len
    assert 21 == chr2_len


def test_get_segm_length():
    gene_names = np.array(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', "G7", "G8", "G9"])

    cns = CNSegmenter(gene_names=gene_names,
                      gene_starts=np.array([0, 5, 10, 15, 20, 25, 0, 6, 15]),
                      gene_ends=np.array([4, 9, 14, 19, 24, 29, 6, 12, 21]),
                      gene_chroms=["chr1"] * 6 + ["chr2"] * 3)

    seg1_len = cns.get_segment_length(['G1', 'G2', 'G3'])
    seg2_len = cns.get_segment_length(['G1', 'G5'])

    assert 14 == seg1_len
    assert 24 == seg2_len

    correct_behaviour = False

    try:
        seg3_len = cns.get_segment_length(['G1', 'G7'])

    except IOError:
        correct_behaviour = True

    assert correct_behaviour


def test_from_gtf():
    #gtf_file = 'example_gtf'

    gtf_file = '/home/mlarmuse/Downloads/Homo_sapiens.GRCh37.75.gtf.gz'
    cns = CNSegmenter.from_gtf(gtf_file)

    print(cns.unique_chroms)

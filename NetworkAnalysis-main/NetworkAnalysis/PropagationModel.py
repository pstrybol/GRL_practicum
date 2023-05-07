import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.preprocessing import normalize, scale

class PropagationModel:

    def __init__(self, layers=1, mode='conditional', causal=True):

        self.mode = mode
        self.layers = layers
        self.causal = causal
        self.Weight = None
        self.genes = None

    def setWeights(self, genomic_data, transcript_data):
        self.Weight = get_weights(genomic_data, transcript_data, layers=self.layers, mode=self.mode, reverse=not self.causal)
        # print(self.Weight.shape)

    def setRiskProfile(self, genomic_data, scale_method='l2'):
        if self.layers == 1:
            RiskProfile = np.matmul(genomic_data, self.Weight)

        else:
            Weight1, Weight2 = self.Weight
            RiskProfile = np.matmul(np.matmul(genomic_data, Weight1), Weight2)

        if scale_method == 'l1':
            RiskProfile = normalize(RiskProfile, axis=1, norm='l1')
        elif scale_method == 'sd':
            RiskProfile = scale(RiskProfile, axis=1)
        elif scale_method == 'l2':
            RiskProfile = normalize(RiskProfile, axis=1, norm='l2')
        elif scale_method == 'max':
            RiskProfile = normalize(RiskProfile, axis=1, norm='max')

        return RiskProfile

    def predictMatch(self, genomic_data, transcript_data, match_method='dot_positives', topn=None, scale_method='l2'):
        RiskProfile = self.setRiskProfile(genomic_data, scale_method=scale_method)
        Match = match_profiles(RiskProfile, transcript_data, method=match_method, topn=topn)

        return RiskProfile, Match

    def IdentifyDrivers(self, expressionData, topn=None, average=True):
        if topn:
                topn_values = np.sort(self.Weight, axis=0)[topn, :]
                mask = 1*(self.Weight >= topn_values)
                Gene_scores = np.matmul(expressionData, np.transpose(self.Weight*mask))
        else:
            Gene_scores = np.matmul(expressionData, self.Weight)

        if average:
            AvgGeneScore = np.mean(Gene_scores, axis=0)
            return AvgGeneScore
        else:
            return Gene_scores

    def getWeights(self):
        return self.Weight

    # TODO: implement these getter and setter functions


# important functions related to generation of weights
def get_logsig_weights(cna_mat, gx_mat=None, testtype='right'):
    # calculates the p-value of the CNV for pairs of genes
    # The input is a N_patient x N genes matrix
    # The function outputs a pval df for every pair significant above pvals_thresh
    # The calculations are very fast, but also very RAM intensive
    # Only includes right-tailed test and left test separately
    N_patients = cna_mat.shape[0]

    if gx_mat is None:
        cna_cna_mat = np.matmul(np.transpose(cna_mat), cna_mat)
        p_cna = np.sum(cna_mat, axis=0)/N_patients
        p = np.outer(p_cna, p_cna)
    else:
        cna_cna_mat = np.matmul(np.transpose(cna_mat), gx_mat)
        p_cna, p_gx = np.sum(cna_mat, axis=0)/N_patients, np.sum(gx_mat, axis=0)/N_patients
        p = np.outer(p_cna, p_gx)

    if testtype.lower() == 'left':
        pvals_mat = binom.logcdf(cna_cna_mat, N_patients, p)
    else:
        pvals_mat = binom.logsf(cna_cna_mat, N_patients, p)

    pvals_mat[np.isinf(pvals_mat)] = 0

    return -1*pvals_mat


def get_conditional_weights(mat, mat2=None, net=None, alpha_prior=0.2):
    # This functions creates weights for network propagation based on conditional probabilities
    # The resulting matrix needs can be right multiplied to the data matrix to propagate
    if mat2 is not None:
        Ngenes = mat2.shape[1]
    else:
        Ngenes = mat.shape[1]

    if mat2 is not None:
        cooc_mat = np.matmul(np.transpose(mat), mat2)
    else:
        cooc_mat = np.matmul(np.transpose(mat), mat)

    # cooc_mat[cooc_mat < 5] = 0 #change this line
    occ_mat = np.transpose(np.tile(np.sum(mat, 0), (Ngenes, 1))+0.00001)

    if net is not None: #Currently not supported
        cooc_mat *= net
        prior_net = alpha_prior * (1-net) + alpha_prior * net
        occ_mat *= prior_net

    return cooc_mat/occ_mat

def get_informationGain(mat, mat2=None, net=None, alpha_prior=0.2):
    # This functions creates weights for network propagation based on conditional probabilities
    # The resulting matrix needs can be right multiplied to the data matrix to propagate
    if mat2 is not None:
        Ngenes = mat2.shape[1]
    else:
        Ngenes = mat.shape[1]

    if mat2 is not None:
        cooc_mat = np.matmul(np.transpose(mat), mat2)
        occ2_mat = np.tile(np.sum(mat2, 0)/mat2.shape[0], (mat.shape[1], 1))
    else:
        cooc_mat = np.matmul(np.transpose(mat), mat)
        occ2_mat = np.tile(np.sum(mat, 0)/np.sum(mat), (Ngenes, 1))

    occ_mat = np.transpose(np.tile(np.sum(mat, 0), (Ngenes, 1))+1)

    if net is not None: #Currently not supported
        cooc_mat *= net
        prior_net = alpha_prior * (1-net) + alpha_prior * net
        occ_mat *= prior_net

    return (cooc_mat+1)/occ_mat - occ2_mat

def get_weights_1layer(genomic_data, transcript_data, mode='conditional', reverse=False, sign_level=np.log(1e-10)):
    if reverse:
        if mode == 'conditional':
            cna_weights = get_conditional_weights(transcript_data, genomic_data)
        elif mode == 'sign-conditional':
            cna_weights = get_conditional_weights(transcript_data, genomic_data) * \
                          ((-1.*get_logsig_weights(transcript_data, genomic_data)) < sign_level)
        else:
            cna_weights = get_logsig_weights(transcript_data, genomic_data)
    else:
        if mode == 'conditional':
            cna_weights = get_conditional_weights(genomic_data, transcript_data)
        elif mode == 'sign-conditional':
            cna_weights = get_conditional_weights(genomic_data, transcript_data) * \
                          ((-1.*get_logsig_weights(genomic_data, transcript_data)) < sign_level)
        elif mode == 'Information Gain':
            cna_weights = get_informationGain(genomic_data, transcript_data)
        else:
            cna_weights = get_logsig_weights(genomic_data, transcript_data)

    return cna_weights


def get_weights_2layer(genomic_data, transcript_data, mode='conditional', clusterlist=None, activation_thresh=0.5):

    if clusterlist is None:
        if mode == 'conditional':
            cna_weights = get_conditional_weights(genomic_data, transcript_data)
            cna_weights = np.triu(np.tril(cna_weights))
            gx_weights = get_conditional_weights(transcript_data)
        else:
            cna_weights = get_logsig_weights(genomic_data, transcript_data)
            cna_weights = np.triu(np.tril(cna_weights))
            gx_weights = get_logsig_weights(transcript_data)

    else:
        cna_weights = get_clustermat(genomic_data, clusterlist, activation_thresh=activation_thresh)
        gx_weights = get_weights_1layer(cna_weights, transcript_data, mode=mode)

    return cna_weights, gx_weights


def get_weights(genomic_data, transcript_data, layers=1, mode='conditional', reverse=False):
    if layers == 1:
        return get_weights_1layer(genomic_data, transcript_data, mode=mode, reverse=reverse)
    else:
        return get_weights_2layer(genomic_data, transcript_data, mode=mode)


def get_clustermat(dataset, clusterlist, activation_thresh=0.5):
    # Helper function to get a participating matrix (binary) indicating which gene belongs to which cluster
    cluster_mat = 1. * np.array([dataset[cluster].sum(axis=1) > (activation_thresh * len(cluster)) for cluster in clusterlist])
    return cluster_mat

def get_topgenes(gene_ids, risk_profile, ntop=1000):
     return gene_ids[np.argsort(-1*np.mean(risk_profile, 0), )][:ntop]

def match_profiles(risk_profile, df_check, method='dot_positives', topn=None):
    #Match risk profiles to the observed data using different measures
    SMALL_NUMBER = 1e-15

    if method == 'dot_positives':
        fit_score = np.diag(np.matmul(risk_profile, np.transpose(df_check)))
    elif method == 'topn':
        idx = np.argsort(-1.*risk_profile, axis=1)
        fit_score = np.array([np.dot(normalize_vector(risk_profile[i, idx[i, :topn]]),
                                     np.transpose(df_check.iloc[i, idx[i, :topn]])) for i in range(risk_profile.shape[0])])
    elif method == 'dot_negatives':
        fit_score = -1. * np.diag(np.matmul(risk_profile, 1 - np.transpose(df_check)))
    else:
        fit_score = logloss(risk_profile, df_check)

    return fit_score

def normalize_vector(v):
    vnorm = np.linalg.norm(v)
    if vnorm == 0:
        u = v
    else:
        u = v/vnorm

    return u

def logloss(rp, df_check, SMALL_NUMBER=1e-15):
    loss = np.diag(np.matmul(np.log(SMALL_NUMBER + rp), np.transpose(df_check))) \
            + np.diag(np.matmul(np.log(1. - rp - SMALL_NUMBER), 1 - np.transpose(df_check)))

    return loss

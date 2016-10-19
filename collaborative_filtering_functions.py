# -*- coding: utf-8 -*-
"""
Collaborative Filtering Functions
Author: flexDW
"""

#%% calculate similarities for a matrix

def adj_cosine_sim(sparse_x):
    """
    Returns the row wise similarity using adjusted cosine.
    """
    
    # create indicator matrix
    binary = sparse_x > 0

    # center non-zero elements only
    col_means = sparse_x.sum(axis=0) / binary.sum(axis=0)
    centering_mat = binary.multiply(sparse.csc_matrix(col_means))
    centered = np.subtract(sparse_x, centering_mat)
    
    # return cosine similarity of centered data
    return cosine_similarity(centered)


#%% collaborative filtering

def collab_filter(test_pairs, train_mat, filter_by, k, min_k=3):
    """
    Returns predicted values using collaborative filtering.
    Arguments:
    - test_pairs: test set values to be predicted with column names 'user' and 'movie'
    - train_mat: the training set user-movie ratings matrix
    - filter_by: either 'user' or 'movie'
    - k: number of neighbors to use in predictions
    - min_k: the minimum number of neighbors for a prediction
 
    Comment: Rows of train_mat should match filter_by so that the 
    number of rows matches the dimensions of sim_mat.
    """
    
    # create predictions array
    n = len(test_pairs)
    preds = np.zeros(n)
    
    # create row-similarity matrix
    sim_mat = adj_cosine_sim(train_mat)

    # set reference column opposite to filter_by column
    if filter_by == 'user': ref_col = 'movie'
    if filter_by == 'movie': ref_col = 'user'

    # exclude users or items left out of training set from predictions
    missing_ref_id = np.setdiff1d(test_pairs[ref_col], train[ref_col])
    missing_filter_id = np.setdiff1d(test_pairs[filter_by], train[filter_by])
    missing_index = np.logical_or(test_pairs[filter_by].isin(missing_filter_id), test_pairs[ref_col].isin(missing_ref_id))
    index = np.arange(n)[np.where(np.invert(missing_index))]
    
    # iterate through predictions
    for i in index:
        # obtain user-item pair for predicting
        filter_id = test_pairs[filter_by].iloc[i]
        ref_id = test_pairs[ref_col].iloc[i]
        
        # get the full ratings vector from the sparse matrix
        rating_vec = np.asarray(train_mat[:, ref_id].todense()).flatten()
        
        # obtain similarity where rating exists, reduce rating vec
        weight_vec = np.asarray(sim_mat[np.nonzero(rating_vec), filter_id]).flatten()
        rating_vec = rating_vec[np.nonzero(rating_vec)]
        weighted_rating = np.multiply(rating_vec, weight_vec)
        n_neighbors = np.sum(weighted_rating > 0)
        end = min(k, np.sum(n_neighbors > 0)) + 1
        if end > min_k:
            idx = np.argsort(weight_vec)[:-end:-1]
            preds[i] = np.sum(np.multiply(weighted_rating[idx])) / np.sum(np.abs(weight_vec[idx]))

    return(preds)

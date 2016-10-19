# Movie Recommendations

This script generates movie recommendations using matrix factorization, collaborative filtering and linear models and an ensemble prediction. The data used is the MovieLens 1M Data Set, 1 million movie predictions from around 6000 users for around 4000 movies - http://grouplens.org/datasets/movielens/1m/. There are 3 data files available in the download but we only use the ratings.dat file. 

The 5-fold cross validation results for each model are below:


For matrix factorization, I borrowed the optimal number of factors and learning rate from MyMedialite and increased the regularization a little. So I would say that these predictions are more or less optimal. However, I think the predictions from collaborative filtering could be improved by increasing the number of neighbours and also experimenting with other similarity measures - a quick play around with correlation showed me that it is as good as adjusted-cosine and possibly one is better for user-user filtering and the other for item-item filtering. 

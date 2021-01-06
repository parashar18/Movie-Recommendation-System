# -*- coding: utf-8 -*-
"""

@author: Parashar Parikh
"""

import pandas as pd
import numpy as np
import sys
from sklearn.metrics import r2_score



def Collabrative_filter(train,test):
    train=pd.read_csv(train)
    test=pd.read_csv(test)

    train.columns = ['movieid', 'userid', 'rating']
    test.columns = ['movieid', 'userid', 'rating']

    print ('Corrilatoin matrix')
    train2 = train.pivot(index = 'userid', columns = 'movieid',values = 'rating')
    user_mean = train2.mean(axis = 1)
    cor_matrix = train2.T.corr().fillna(0)
    print(cor_matrix)
    
    
    rating_diff = train2.sub(train2.mean(axis=1), axis=0)
    user_mean_of_means = user_mean.mean()
    users_all = user_mean.index
    movies_all = rating_diff.columns
    predictions = []

    for i in range(len(test)):
        movie_j, user_a, = [int(test.iat[i,0]), int(test.iat[i,1])]
        if user_a in users_all:
            user_a_mean = user_mean.loc[user_a]
        else:
            user_a_mean = user_mean_of_means
        if movie_j in movies_all:
            rating_diff_js = rating_diff.loc[:,movie_j]
            users_a_no_k = rating_diff_js[rating_diff_js.isnull()].index
            w = cor_matrix.loc[user_a,:].copy()
            w.loc[users_a_no_k] = None
            sum_w = w.abs().sum()
            if sum_w != 0:
                coll_rating = (w * rating_diff_js).sum() / sum_w
            else:
                coll_rating = 0
        else:
            coll_rating = 0
        rating_ik = user_a_mean + coll_rating
        predictions.append( round(max(min(rating_ik, 5),1),1) )

    
    test['predictions'] = predictions
    RMSE=round(np.sqrt( ((test['rating'] -test['predictions'])**2).mean()),5)
    MAE=round(np.mean(abs(test['rating'] -test['predictions'])),5)
    r2score=r2_score(test['rating'],test['predictions'])
     
    
    return RMSE,MAE,r2score 
if __name__ == "__main__":
    train=sys.argv[1]
    test=sys.argv[2]
    RMSE,MAE,r2score=Collabrative_filter(train,test)
    print("RMSE:", RMSE)
    print("MAE:", MAE)
    print("R2_Score:",r2score)
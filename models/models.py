import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import xgboost as xgb

class LGBMRegressor:
    """
        LGBMRegressor model part
    """
    def __init__(self,pred_rank_all,eval_num,target,params=None):
        self.model = None
        self.train_on_alldata = False
        self.using_time_to_split = True
        self.pred_rank_all = pred_rank_all
        self.eval_num = eval_num
        self.target = target
        self.params = {
            'params': {"objective": "regression",                       
                       "metric": "l2_root", 
                       'verbosity': -1,
                       "seed": 0, 
                       'two_round': False,
                       'num_leaves': 64, 
                       'learning_rate': 0.09,
                       'bagging_fraction': 0.9, 
                       'bagging_freq': 3,
                       'feature_fraction': 0.9,
                       'n_estimators':2000,
#                       'n_estimators':50,
                       'min_sum_hessian_in_leaf': 0.1,
                       'lambda_l1': 0.5,
                       'lambda_l2': 0.5,
#                       'min_data_in_leaf': 50,
                       },
#            'early_stopping_rounds': 100,
#            'num_boost_round': 20000,
            'verbose_eval': 20
        }
        if params is not None:
            self.params = params

    def fit(self, X_train, y_train, categorical_feature=None, X_eval=None, y_eval=None):

        
        if X_eval is None or y_eval is None:
            
            if self.using_time_to_split:               
                # when using time to split the dataset
                print("bug?")
                time_delta = (max(X_train.time_rank)-int(self.pred_rank_all/self.eval_num))
                print(time_delta,"####")
                X_tr = X_train[X_train.time_rank<time_delta]
                X_eval = X_train[X_train.time_rank>=time_delta]
                y_tr = y_train[X_train.time_rank<time_delta]
                y_eval = y_train[X_train.time_rank>=time_delta]

                X_tr = X_tr.drop(["time_rank"],axis=1)
                X_eval = X_eval.drop(["time_rank"],axis=1)

#                X_train.drop(["time_rank"],axis=1,inplace =True)
#                X_eval.drop(["time_rank"],axis=1,inplace =True)
        
            else:
                
                X_tr, X_eval, y_tr, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
                
            print(X_eval["uid"].value_counts())
            print(X_tr["uid"].value_counts())
                
            self.feature_name = list(X_tr.columns)
            print(X_tr.shape,X_eval.shape)
            print(y_tr.shape,y_eval.shape)
        
        
#        lgb_train = lgb.Dataset(X_tr, y_tr)
#        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

#        self.model = lgb.train(train_set=lgb_train, valid_sets=lgb_eval, valid_names='eval', **self.params,
#                               early_stopping_rounds=100)

        c = []
        print(self.feature_name)
        print(self.target)
        for i,col in enumerate(self.feature_name):
            if col in self.target+["month",'day',"hour"]:
                c.append(i)   
        print(c)
        self.model = CatBoostRegressor(iterations=1000,
                                     learning_rate=0.3,
                                     depth=10,
                                     eval_metric='RMSE',
                                     random_seed = 42,
                                     bagging_temperature = 0.2,
                                     od_type='Iter',
                                     metric_period = 50,
                                     cat_features=c,
                                     od_wait=20)
        self.model = self.model.fit(X_tr, y_tr,eval_set=(X_eval, y_eval),use_best_model=True,verbose=50)
        
        if self.train_on_alldata:
            print("train on all data")
            self.params['params']['n_estimators'] = int(self.model.best_iteration*1.0*1.4)
            lgb_train = lgb.Dataset(X_train, y_train)

            self.model = lgb.train(train_set=lgb_train, valid_sets=None, **self.params)

        return self

    
    def predict(self, X_test):
        
        if self.model is None:
            raise ValueError("You must fit first!")

        return self.model.predict(X_test)

    
    def score(self):
        """
            we sort the dict to show which feature is more important.
        """
        
#        dic = dict(zip(self.feature_name, self.model.feature_importance('gain')))
        
#        return dict(sorted(dic.items(), key=lambda d: d[1]))
        return 1


import os
os.system('pip install catboost')
os.system('pip install statsmodels')
os.system("python -m pip install --upgrade scikit-learn==0.19.2")
import pickle
import pandas as pd
import numpy as np
from models import LGBMRegressor
from preprocessing import parse_time, TypeAdapter,Extract_features,Get_train,add_statistics
from preprocessing import lag_year,lag_month,daily_feature,add_cyc,del_with_num,del_with_top5
#os.system("pip3 install '' --force-reinstall")

import sklearn
print(sklearn.__version__)
import pkg_resources
pkg_resources.require("scikit-learn==0.19.2")  # modified to use specific numpy
print(sklearn.__version__)
import math
import sklearn


class Model:
    def __init__(self, info, test_timestamp, pred_timestamp):
        self.info = info
        self.primary_timestamp = info['primary_timestamp']
        self.primary_id = info['primary_id']
        self.max_y = 0
        self.un_primary = len(self.primary_id)==0
        
        target = ["uid"]
#        target = []
        if len(self.primary_id)>1:
            self.target = self.primary_id + target
        elif len(self.primary_id)==0:
            self.target = []
        else:
            self.target = target
        
        
        self.label = info['label']   # self.label = "A4"
        self.schema = info['schema']
        self.pred = 0
        self.count = 0     # use for fit part
        

        print(f"\ninfo: {self.info}")

        self.dtype_cols = {}
        self.dtype_cols['cat'] = [col for col, types in self.schema.items() if types == 'str']        
        self.dtype_cols['num'] = [col for col, types in self.schema.items() if types == 'num']
        
        
        self.dtype_cols['num_no_label'] = [col for col in self.dtype_cols['num'] if col not in [self.label]]
        self.dtype_cols['str_no_priamry_id'] = [col for col in self.dtype_cols['cat'] if col not in self.primary_id]
        print("##")
        print(self.dtype_cols['num_no_label'],self.dtype_cols['str_no_priamry_id'],sep='\t')
        self.test_timestamp = test_timestamp
        self.pred_timestamp = pred_timestamp


        self.n_test_timestamp = len(pred_timestamp)  # 365
        
        ## warning!整除部分暂时无法对齐会报错

        if self.n_test_timestamp <=50:
            num = 5
            self.update_c = 2
        elif 50<self.n_test_timestamp<= 300:
            num = 5
            self.update_c = 3
        elif 300<self.n_test_timestamp <= 1000:
            num = 5
            self.update_c = 3
        elif 1000<self.n_test_timestamp:
            num = 5
            self.update_c = 3
            
        nums = []
        while(num > self.n_test_timestamp):
            num -= 2
            
        if self.n_test_timestamp % num ==0:
            for i in range(1,5):
                temp = num - i
                if self.n_test_timestamp % temp !=0:
                    nums.append(temp)
            num = max(nums)
       
        print(f"in this dataset we update {num} times")
        
        
        self.update_interval = math.ceil(self.n_test_timestamp / num)
        #更新的时间戳
        
        print("self.update_interval = ",self.update_interval)
        
        
        if self.n_test_timestamp % num == 0:
            self.update_n = self.update_interval
        else:
            self.update_n = self.update_interval - 1
            
            
        print(self.update_interval)

        print(f"sample of test record: {len(test_timestamp)}")
        print(f"number of pred timestamp: {len(pred_timestamp)}")
        
        
        self.lgb_model = LGBMRegressor(self.n_test_timestamp,self.update_c,self.target)
        self.n_predict = 0
        self.extract = Extract_features(self.update_interval,self.label,self.primary_timestamp,self.n_test_timestamp)
        print(f"Finish init\n")

        
    def train(self, train_data, time_info):
        """
            type_adapter : get uid & deal type
            extarct      : extract_features
        """
        ## 
        print("begin train")
        
        ## 第一次训练
        if self.count ==0:
            self.type_adapter = TypeAdapter(self.primary_id)  # 输入primary_id 转化为uid使用
            
##            if len(self.dtype_cols['num_no_label'])>=2:
##                self.top5num,self.usefull_num = del_with_num(train_data,self.dtype_cols['num_no_label'],self.label)
                
##                train_data = del_with_top5(train_data,self.top5num)
                # 确认usefull_num 和top5num
            
            
            
            train_data = self.type_adapter.fit_transform(train_data)   # get the uid
            
            X = train_data.copy()
            X = self.extract.get_rank(X)            # get rank
            self.max_rank = max(X["time_rank"])     # 

            X = X.drop_duplicates().sort_values(by=["uid",self.primary_timestamp]).reset_index(drop=True)

            # get data for feature
            train_temp = X.copy()
            train_temp["time_rank"] += self.update_interval
            train_temp = train_temp[train_temp.time_rank>max(X.time_rank)].copy()
            data = pd.concat([X,train_temp],axis=0)        
            data = data[["uid","time_rank",self.label]]
            data = data.drop_duplicates().sort_values(by=["uid","time_rank"]).reset_index(drop=True)

            
            X = self.extract.train_part(data,X) 

            # parse time feature
            time_fea = parse_time(X[self.primary_timestamp])
            X = pd.concat([X, time_fea], axis=1)
#           
#            X,self.cyc_dict = add_cyc(X, train = True)

            
#            self.target_fea = daily_feature(self.target,X,self.label)
#            for i in self.target_fea.keys():
#                for j in self.target_fea[i].keys():
#                    X = pd.merge(X,self.target_fea[i][j],on = [str(i),str(j)])
#                X[col+'_avg_label'] = X[col+'_avg_label'].values + np.random.normal(scale=1.6,size=(len(X),))
                
                
#    test = pd.merge(test,month_fea[col],on=[col,"month"])            
            # add statistics
#            if len(self.dtype_cols['num_no_label']) !=0:
#                X = add_statistics(X,self.dtype_cols['num_no_label'])

            # add month_lag & year_lag

            # drop nan & other feature
        
        
        # 隔update_c 次进行一次训练,其余跳过训练。
        if self.count % self.update_c ==0:
            
            # 当不是第一次输入的数据时，直接截取数据进行训练
            if self.count != 0:   
                
                X = train_data
            
            y = X[self.label].copy()   
            X = X.drop([self.label,self.primary_timestamp],axis=1)  

            # change time rank      ,"time_rank"                          
            print("begin fit")
            self.lgb_model.fit(X, y)            

#            print(X["uid"].value_counts())
            print(f"Feature importance: {self.lgb_model.score()}")
        # 每次update完以后,重置self.predict = 0
        self.predict_num = 0                        
        self.count += 1
                
        print("Finish train\n")

        next_step = 'predict'
        return next_step

    
    def predict(self, new_history, pred_record, time_info):
        if self.n_predict % 100 == 0:
            print("predcit begun!!!!!!")
                
        self.n_predict += 1

        # type adapter
        pred_record = self.type_adapter.transform(pred_record)
        
##        if len(self.dtype_cols['num_no_label'])>=2:
##            pred_record = del_with_top5(pred_record,self.top5num)
        
        pred_record = self.extract.pred_get_rank(pred_record)
#        pred_record["rank_yu"] = pred_record["time_rank"]
        
        
#        if self.predict_num <=1:
#            print(pred_record.head())        
        # lag_feature 12/5
        pred_record = self.extract.predict_part(pred_record)
        
        #当不是第一次pred的时候,我们使用上一次的预测结果去填充lag1. 12/15
#        if self.predict_num !=0:
#            print("we use predictions to lag1")
#            print(len(pred_record),len(self.predictions_values))
#            print(self.predictions_values)
#            pred_record[self.label + "_lag_1"] = self.predictions_values
        
        
        # parse time feature
        time_fea = parse_time(pred_record[self.primary_timestamp])
        pred_record = pd.concat([pred_record, time_fea], axis=1)
#        pred_record = add_cyc(pred_record, train = False,di=self.cyc_dict)
        


#        for i in self.target_fea.keys():
#            for j in self.target_fea[i].keys():
#                pred_record = pd.merge(pred_record,self.target_fea[i][j],on = [str(i),str(j)])
                
#            pred_record[col+'_avg_label'] = pred_record[col+'_avg_label'].values + np.random.normal(scale=1.6,size=(len(pred_record),))
        # add statistics
#        if len(self.dtype_cols['num_no_label']) !=0:
#            pred_record = add_statistics(pred_record,self.dtype_cols['num_no_label'])
        
                    
        pred_record.drop([self.primary_timestamp,"time_rank"], axis=1, inplace=True)
          
        predictions = self.lgb_model.predict(pred_record)
        
        
        #save the predictions for lag1
        #change the self.predict
        
#        self.predictions_values = predictions
#        self.predict_num += 1

#        print(pred_record)
#        if self.n_predict > self.update_interval:  # change ori > to >= 
#        change self.update_interval to update_n:

        if self.n_predict > self.update_n:
            next_step = 'update'
            self.n_predict = 0

        else:
            next_step = 'predict'

        return list(predictions), next_step

    
    def update(self, train_data, test_history_data, time_info):
        print(f"\nUpdate time budget: {time_info['update']}s")

        total_data = pd.concat([train_data, test_history_data])
              
        print(f"更新了{test_history_data[self.primary_timestamp].nunique()}个时间戳的数据")
              
        print(test_history_data.head(2))   
        self.history_timestamp = test_history_data[self.primary_timestamp].nunique()
        ## move from train_part
              
#        total_data = total_data.reset_index(drop=True)
        total_data = self.type_adapter.fit_transform(total_data)   # get the uid
              
##        if len(self.dtype_cols['num_no_label'])>=2:
##            total_data = del_with_top5(total_data,self.top5num)
              
#        print(total_data[total_data["uid"]==total_data.loc[1,"uid"]])
#        total_data = Get_train(total_data,self.primary_timestamp,self.label)

#        train_data[self.label] = np.log1p(train_data[self.label])

        X = total_data.copy()
        X = self.extract.get_rank(X)            # get rank
        self.max_rank = max(X["time_rank"])

        X = X.drop_duplicates().sort_values(by=["uid",self.primary_timestamp]).reset_index(drop=True)

        # get data for feature
        train_temp = X.copy()

        train_temp["time_rank"] += self.update_interval

        train_temp = train_temp[train_temp.time_rank>max(X.time_rank)].copy()
        data = pd.concat([X,train_temp],axis=0)        
        data = data[["uid","time_rank",self.label]]
        data = data.drop_duplicates().sort_values(by=["uid","time_rank"]).reset_index(drop=True)

        X = self.extract.train_part(data,X) 

        # parse time feature
        time_fea = parse_time(X[self.primary_timestamp])
        X = pd.concat([X, time_fea], axis=1)
#        X,self.cyc_dict = add_cyc(X, train = True)
              
#        self.target_fea = daily_feature(self.target,X,self.label)
#        for i in self.target_fea.keys():
#            for j in self.target_fea[i].keys():
#                X = pd.merge(X,self.target_fea[i][j],on = [str(i),str(j)])             
#            X[col+'_avg_label'] = X[col+'_avg_label'].values + np.random.normal(scale=1.6,size=(len(X),))
              # add statistics
#        if len(self.dtype_cols['num_no_label']) !=0:
#            X = add_statistics(X,self.dtype_cols['num_no_label'])

        # add month_lag & year_lag
              
        self.train(X, time_info)

        print("Finish update\n")

        next_step = 'predict'
        return next_step

              
    def save(self, model_dir, time_info):
        print(f"\nSave time budget: {time_info['save']}s")

        pkl_list = []

        for attr in dir(self):
            if attr.startswith('__') or attr in ['train', 'predict', 'update', 'save', 'load']:
                continue

            pkl_list.append(attr)
            pickle.dump(getattr(self, attr), open(os.path.join(model_dir, f'{attr}.pkl'), 'wb'))

        pickle.dump(pkl_list, open(os.path.join(model_dir, f'pkl_list.pkl'), 'wb'))

        print("Finish save\n")


    def load(self, model_dir, time_info):
        print(f"\nLoad time budget: {time_info['load']}s")

        pkl_list = pickle.load(open(os.path.join(model_dir, 'pkl_list.pkl'), 'rb'))

        for attr in pkl_list:
            setattr(self, attr, pickle.load(open(os.path.join(model_dir, f'{attr}.pkl'), 'rb')))

        print("Finish load\n")
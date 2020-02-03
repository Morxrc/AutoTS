import pandas as pd
import numpy as np
pd.options.display.max_columns = None
import heapq
from statsmodels.tsa.stattools import acf, pacf
import lightgbm as lgb

def stats_acf(x, unbiased=False, nlags=40, qstat=False, fft=None, alpha=None,
        missing='none'):
    """
    Calculate the autocorrelation function.
    Parameters
    ----------
    x : array_like
       The time series data.
    unbiased : bool
       If True, then denominators for autocovariance are n-k, otherwise n.
    nlags : int, optional
        Number of lags to return autocorrelation for.
    qstat : bool, optional
        If True, returns the Ljung-Box q statistic for each autocorrelation
        coefficient.  See q_stat for more information.
    fft : bool, optional
        If True, computes the ACF via FFT.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett's formula.
    missing : str, optional
        A string in ['none', 'raise', 'conservative', 'drop'] specifying how the NaNs
        are to be treated.
    Returns
    -------
    """
    unbiased = bool_like(unbiased, 'unbiased')
    nlags = int_like(nlags, 'nlags')
    qstat = bool_like(qstat, 'qstat')
    fft = bool_like(fft, 'fft', optional=True)
    alpha = float_like(alpha, 'alpha', optional=True)
    missing = string_like(missing, 'missing',
                          options=('none', 'raise', 'conservative', 'drop'))

    if fft is None:
        import warnings
        warnings.warn(
            'fft=True will become the default in a future version of '
            'statsmodels. To suppress this warning, explicitly set '
            'fft=False.',
            FutureWarning
        )
        fft = False
    x = array_like(x, 'x')
    nobs = len(x)  # TODO: should this shrink for missing='drop' and NaNs in x?
    avf = acovf(x, unbiased=unbiased, demean=True, fft=fft, missing=missing)
    acf = avf[:nlags + 1] / avf[0]
    if not (qstat or alpha):
        return acf
    if alpha is not None:
        varacf = np.ones(nlags + 1) / nobs
        varacf[0] = 0
        varacf[1] = 1. / nobs
        varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1]**2)
        interval = stats.norm.ppf(1 - alpha / 2.) * np.sqrt(varacf)
        confint = np.array(lzip(acf - interval, acf + interval))
        if not qstat:
            return acf, confint

    if qstat:
        qstat, pvalue = q_stat(acf[1:], nobs=nobs)  # drop lag 0
        if alpha is not None:
            return acf, confint, qstat, pvalue
        else:
            return acf, qstat, pvalue


def del_with_num(df,nums,label):
    
    params = {
        'params': {"objective": "regression",
                   "metric": "l2", 
                   'verbosity': -1,
                   "seed": 0,
                   'num_leaves': 64, 
                   'learning_rate': 0.1, 
                   'bagging_fraction': 0.8, 
                   'bagging_freq': 3,
                   'feature_fraction': 0.8,
                   'min_sum_hessian_in_leaf': 0.1,
                   'n_estimators':1000,
                   'lambda_l1': 0.5,
                   'lambda_l2': 0.5,
                   "depth":-1,
                   },
        'verbose_eval': 20
    }
    print("num feature select!")
    lgb_train = lgb.Dataset(df[nums], df[label])

    model = lgb.train(train_set=lgb_train, **params)
    
    feature_importance  = pd.DataFrame()
    feature_importance["feature_names"] = df[nums].columns
    feature_importance["feature_importance"] = model.feature_importance()
    
    if len(nums)>=5:
        top5num = feature_importance["feature_names"][:5].tolist()
    else:
        top5num = feature_importance["feature_names"][:2].tolist()
    
    usefull_num = feature_importance[feature_importance!=0]["feature_names"].tolist()
    
    
    print("feature select down! result = ")
    print(top5num,usefull_num)
    return top5num,usefull_num
    
    
def del_with_top5(df,top5num):
    """
        simple engeneering for top5num
        with * / - log
    """
    for i in range(len(top5num)-1):
        for j in range(i,len(top5num)):
            df[f"{i}/{j}"] = df[top5num[i]].astype(float) / df[top5num[j]].astype(float)
            df[f"{i}_{j}"] = df[top5num[i]].astype(float) - df[top5num[j]].astype(float)
            df[f"{i}*{j}"] = df[top5num[i]].astype(float) * df[top5num[j]].astype(float)
            df[f"log{i}"] = np.log1p(df[top5num[i]].astype(float))
            if i == 3:
                df[f"log{i+1}"] = np.log1p(df[top5num[i+1]].astype(float))
    return df
     
    

def parse_time(xtime: pd.Series):
    """
        time_feature:
        input - pd.Series()
        output - pd.DataFrame()
    """
    result = pd.DataFrame()
    dtcol = pd.to_datetime(xtime, unit='s')
    
    result['ori'] = dtcol.astype('int64') // 10**9
    result['year'] = dtcol.dt.year
    result['month'] = dtcol.dt.month
    result['day'] = dtcol.dt.day
    result["dayofweek"] = dtcol.dt.dayofweek
    result["isweekend"] = dtcol.dt.dayofweek // 5
    result["dayofyear"] =  dtcol.dt.dayofyear
    result["weekofyear"] = dtcol.dt.weekofyear
    result['weekday'] = dtcol.dt.weekday
    result['hour'] = dtcol.dt.hour
    return result

#def del_num_feature():
#处理num_feature的part   


def handle_acf(data, k):
    """
    Autocorrelation function
    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)    



def add_cyc(df, train = True,di=None):

    if train:
        dic = {}
        columns = ["month","weekday","hour",'day']   
        for col in columns:
            if df[col].nunique() >=3:
                dic[col] = max(df[col])
                df[col + '_sin'] = np.sin((2*np.pi*df[col])/dic[col])
                df[col + '_cos'] = np.cos((2*np.pi*df[col])/dic[col])
        print(dic)
        return df,dic
    
    else:
        for col in di.keys():
            df[col + '_sin'] = np.sin((2*np.pi*df[col])/di[col])
            df[col + '_cos'] = np.cos((2*np.pi*df[col])/di[col])
        return df
                        

def reduce_mem_usage(df, verbose=True):
    """
        use for reduce memory usage
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
#                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                    df[col] = df[col].astype(np.int16)
#                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
                else:
                    df[col] = df[col].astype(np.int32)
            else:
#                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                    df[col] = df[col].astype(np.float16)
#                else:
                 df[col] = df[col].astype(np.float32)
#                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                    df[col] = df[col].astype(np.float32)
#                else:
#                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def Get_train(df,time,label):
    """
        we notice that a lot of 0 at the begining in both dataset,
        so we use this function to remove that
    """
    time = str(time)
    label = str(label)
    
    df = df.drop_duplicates().sort_values(by=["uid",time]).reset_index(drop=True)
    
    for i in range(10000):
        if df.loc[i,label]!=0:
            stop_time = df.loc[i,time]
            break

    df = df[df[time] >= stop_time]

    return df


def lag_feature(df,lags,col):
    """
        use for lag_feature
    """
    
    tmp = df[["uid","time_rank",col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ["uid","time_rank",col+"_lag_"+str(i)]
#        shifted[col+"_lag_"+str(i)] = shifted[col+"_lag_"+str(i)].values #+ np.random.normal(scale=1.6,size=(len(tmp),))
        shifted["time_rank"] += i
        df = pd.merge(df,shifted,on=["uid","time_rank"],how="left")       
    return df


def lag_month(df,lags,col):
    """
        use for lag_month
        lags [1-11]
    """

    tmp = df[["uid","month",col,"year"]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ["uid","month",col+"_lag_month_"+str(i),"year"]
#        shifted[col+"_lag_"+str(i)] = shifted[col+"_lag_"+str(i)].values #+ np.random.normal(scale=1.6,size=(len(tmp),))
        shifted["month"] += i
        shifted.loc[shifted.month>=13,"year"] += 1
        shifted.loc[shifted.month>=13,"month"] -=12
        df = pd.merge(df,shifted,on=["uid","month","year"],how="left")       
    return df


def lag_year(df,lags,col):
    """
        use for lag_year
        lags [1-11]
    """
    tmp = df[["uid","month",col,"year"]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ["uid","month",col+"_lag_year_"+str(i),"year"]
        shifted["year"] += i
        df = pd.merge(df,shifted,on=["uid","month","year"],how="left")       
    return df



def choose_lags(time_rank,time_all,Series):
    """
        define the shift list
    """
    Series = Series.head(time_all*4)
    Series = Series + np.random.normal(scale=0.3,size=(len(Series),))
#    if febonacci:
#        lags =[1, 2, 3, 5, 8, 13, 21, 34, 55]# 89, 144, 233,377,610]
#        lags = [i+time_rank for i in lags if (i+time_rank< time_all)]
#    lags = [2,4,7,12,14,15,24,28,30,31,60,72,84,112,119,182,360,365,365+7,365+12,365+14,365+30] #,546,728,730]
#    print(Series)
#    np.errstate(invalid='ignore', divide='ignore')
    lag_acf = acf(Series, nlags=(time_all+1))
    print(lag_acf)
#    print(acf(Series, nlags=(time_all+1)))
    temp = lag_acf[time_rank:]
    lags = heapq.nlargest(20,range(len(temp)), temp.take)
    
    lags = [i+time_rank for i in lags]#+[time_all]
    print(lags)    
#    lags = [i for i in lags if (i >=time_rank and i<=time_all)]
    return lags


def add_statistics(data,cols):
    """
        add statistics
    """
    
#    data.replace(np.nan,0,inplace = True)
    data["the_median"] = data[cols].median(axis=1)
    data["the_mean"] = data[cols].mean(axis=1)
    data["the_sum"] = data[cols].sum(axis=1)
    data["the_kur"] = data[cols].kurtosis(axis=1)
    data["the_std"] = data[cols].std(axis=1)
    
    return data
    
#def add_mean_target_encoding(data,cols):


def daily_feature(cats,df,label):
    daily_feature = {}
    for col in ["hour","day","month"]:
        daily_feature[col] = {}
        if df[col].nunique()>10:
            for cat in cats: 
                group = df.groupby([cat,col]).agg({label: ['mean']})
                group.columns = [cat+"_"+col+'_avg_label']
                group.reset_index(inplace=True)
                daily_feature[col][cat] = group
    return daily_feature

#def mean_day(cols,df,label):

def exponential_smoothing(series, alpha):
    """
        series - dataset with timestamps
        alpha - float [0.0, 1.0], smoothing parameter
    """
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

#def feature_select(df,self.label,self.):


#def mean_encoding_for_multiv


def get_diff(df,col,lags):
    for i in range(len(lags)-1):
        df[str(lags[i])+"diff"+str(lags[i+1])] = df[col+"_lag_"+str(lags[i])]/df[col+"_lag_"+str(lags[i+1])]
        df[str(lags[i])+"-"+str(lags[i+1])] = df[col+"_lag_"+str(lags[i])] -df[col+"_lag_"+str(lags[i+1])]
    return df


class Extract_features:
    """
    123
    """
    def __init__(self,pred_time_rank,label,primary_timestamp,pred_time_all):
        
        self.pred_time_rank = pred_time_rank 
        self.label = label
        self.primary_timestamp = primary_timestamp
        self.windows = [5]
        self.pred_time_all = pred_time_all
        self.temp_record = 0
        self.iter_num = 0
        
    def train_part(self,data,X):
                
        # define lags 
        
#        for u in data.uid.unique():
#            data.loc[data.uid==u,self.label] = exponential_smoothing(data.loc[data.uid==u,self.label].values,0.6)

#        data[self.label] = data[self.label].round(0)
        
        if self.iter_num == 0:
            temp4lag = X[X["uid"]==1][self.label]
            lags = choose_lags(self.pred_time_rank,self.pred_time_all,temp4lag)
            self.lags = lags
            self.iter_num = 1
            del temp4lag
        print(self.lags)

        start = max(self.lags)
        
        
        # lag feature
        self.feature = lag_feature(data,self.lags,self.label)
        # del + [1]
        
        # window feature
        for window in self.windows:
            # Min value
            f_min = lambda x: x.rolling(window=window).min()
            exp_min = lambda x: x.expanding().min()
            # Max value
            f_max = lambda x: x.rolling(window=window).max()
            exp_max = lambda x: x.expanding().max()
            # Mean value
            f_mean = lambda x: x.rolling(window=window).mean()
            exp_mean = lambda x: x.expanding().mean()
            # Standard deviation
            f_std = lambda x: x.rolling(window=window).std()

            function_list = [f_min, f_max, f_mean, f_std,]# exp_min, exp_max, exp_mean]
            function_name = ['min', 'max', 'mean', 'std', ]#"exp_min", "exp_max", "exp_mean"]   

            for i in range(len(function_list)):
                for lag in self.lags[0:3]:
                    self.feature[("label_%s"%function_name[i])+"lag_"+str(lag)+"_"+str(window)] = self.feature.groupby(["uid"])[self.label].apply(function_list[i]).shift(lag)

        # end trian, merge the data
        self.feature.drop([self.label],axis=1,inplace=True)
        X = pd.merge(X,self.feature,on=["uid","time_rank"],how="left")
        
        X = X[X.time_rank>(start+max(self.windows)+self.temp_record)]
        
#        X["time_yu"] = X["time_rank"]%(self.pred_time_all+1)
        self.shape = X.shape[1]
#        X = get_diff(X,self.label,self.lags)
        
        print("!!!!!",X.shape)
        self.temp_record += self.pred_time_rank
        
        return X
      
    
    def predict_part(self,test):
        test = pd.merge(test,self.feature,on=["uid","time_rank"],how="left")
        
        if test.shape[1]!=self.shape-1:    
            print("test shape != train.shape")
            print(f"test shape = {test.shape[1]}, train shape = {self.shape}")
            raise 
#        test = get_diff(test,self.label,self.lags)
#        print(test)
#        print("test_merge:!!!",max(self.feature.time_rank))
#        print(test.time_rank)
        return test


    def get_rank(self,df):
        """
            in some datasets, There were some uids which popped up in the middle,
            so we change the get_rank function to handle this problem 
        """
        rank_map = dict(zip(sorted(df[self.primary_timestamp].unique()),range(1,df[self.primary_timestamp].nunique()+1)))
        df["time_rank"] = df[self.primary_timestamp].map(rank_map)
        self.time_rank = max(df.time_rank)
        return df

    
    def pred_get_rank(self,test):
        self.time_rank += 1
        test["time_rank"] = self.time_rank
        
        return test
    
    

class TypeAdapter:
    """
        adapt_cols    —— 传入一个值str列的col list.
                      for example : [primary_id1,primary_id2].
        label_encoder —— dict,对uid进行label_encoder处理.
                      对于某一些数据集, 有一些uid中途加进来，这样对于更新就会错行。
                      因此改回hash_m
    """
    
    def __init__(self, primary_id):
        self.adapt_cols = [] 
        self.label_encoder = {}
        self.primary_id_cols = primary_id
        self.trs_num = 0
        
    def fit_transform(self, X):
         
        if len(self.primary_id_cols)==1:
            X = X.rename(columns={self.primary_id_cols[0]:"uid"})
       
    
        else:
            print("uid_cols:",self.primary_id_cols)
            X["uid"] = "_"
            for i in self.primary_id_cols:
                X["uid"] += "-"+X[i].astype(str)
                
        if self.trs_num == 0:        
            self.label_encoder = dict(zip(X["uid"].unique(),range(X["uid"].nunique())))
            
        else:           
            for i in set(X["uid"].unique()).difference(set(self.label_encoder.keys())):
                print("########",i)
                max_label = max(self.label_encoder.values())
                self.label_encoder[i] = max_label+1
                print("##############",i,max_label+1)
                
        self.trs_num = 1
        
        X["uid"] = X["uid"].map(self.label_encoder)
        
        
        cols_dtype = dict(zip(X.columns, X.dtypes))
        
        # encoding the str using the hash function
        for key, dtype in cols_dtype.items():
            if dtype == np.dtype('object'):# and key not in self.primary_id_cols:
                self.adapt_cols.append(key)
            if key in self.adapt_cols:
                X[key] = X[key].apply(hash_m) 
                
        X = reduce_mem_usage(X,verbose=True)
        print(self.label_encoder)
        return X

    
    def transform(self, X):
                
        if len(self.primary_id_cols) == 1:
            X = X.rename(columns={self.primary_id_cols[0]:"uid"})    
            
        else:
            X["uid"] = "_"
            for i in self.primary_id_cols:
                X["uid"] += '-' + X[i].astype(str)
        
        if X["uid"].map(self.label_encoder).dtype=="int":
            print("###ori")
            X["uid"] = X["uid"].map(self.label_encoder)
            
        else:
            print("update_label_encoder")
            for i in set(X["uid"].unique()).difference(set(self.label_encoder.keys())):
                max_label = max(self.label_encoder.values())
                self.label_encoder[i] = max_label+1
            X["uid"] = X["uid"].map(self.label_encoder)
            
        for key in X.columns:
            if key in self.adapt_cols:
                X[key] = X[key].apply(hash_m)            

                
#        X = reduce_mem_usage(X,verbose=False)  

        return X


def hash_m(x):
    return hash(x) % 1048575

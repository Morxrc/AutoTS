@MeisterMorxrc

# AutoTS 

## 一些关于AutoTimeSeries 问题难点summary:

	1. timeseries 问题是对未来进行预测，对于不同的TS问题，如何选取更加stable的验证集和方法？
	2. 如何从时间序列数据集中的自动提取有效的特征？
	3. 如何处理不同长度(time_diff)的时间序列问题？
	4. 其他autoML问题(主键自动识别,AutoFE,AutoFeatureselect,Hyperparameters Optimization等等)
	5. 如果使用nn 模型的话:如何设计有效神经网络结构
	6. 如果采用GBDT模型的情况下,GBDT学习不到趋势信息,如何解决这个模型本身的问题?


## 本项目工作:

限于memory issues(16 RAM docker was used)本model采用GBDT模型进行建模，具体内容大概如下:

1. FE part：

   **主要code 在models.preprocessing文件中**
   1. most critical numerical features: (Numerical operations (addition, subtraction, multiplication, and division) of pairs of numerical feature) so I use GBDT's gain importance to select the top-n most important numerical features for pairs.
   2. lag feature:
      1. use the acf & pacf to Calculate the autocorrelation function. then select the top-n lag.
      2. last year,month,ect...
   3. time-series feautre: dayofweek,month,dayofyear,isweekend and so on.
   4. unique_key(uid) hash
   5. window feature:
      using the smoothing target to extract window feature

2. reduce_mem_usage part.

3. validation_part:
   train/val split 0.9/0.1 then refit full data with optim epochs

4. model frequent updating part.

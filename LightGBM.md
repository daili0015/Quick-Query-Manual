# LightGBM

## 分割数据

```python
from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=0) 
```

## 参数搜索
[参数意义查询](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
```python
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#31是个神奇的数字
estimator = lgb.LGBMRegressor(num_leaves=31, objective='mse')

param_grid = {
    'learning_rate': [0.07, 0.08],
    'n_estimators': [ 150, 170, 190],
}
# RandomizedSearchCV或者GridSearchCV
gbm = GridSearchCV(estimator, param_grid, cv=3, verbose=2, scoring='neg_mean_squared_error')
gbm.fit(X_train, y_train)

model = gbm.best_estimator_
print('特征重要度:', list(model.feature_importances_))
print('随机搜索-度量记录：',grid.cv_results_)  # 包含每次训练的相关信息
print('随机搜索-最佳度量值:',grid.best_score_)  # 获取最佳度量值
print('随机搜索-最佳参数：',grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('随机搜索-最佳模型：',grid.best_estimator_)  # 获取最佳度量时的分类器模型
```
[参数设定的技巧](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

如果不用网格搜索，可以直接用sklearn风格的接口函数
```
model = lgb.LGBMRegressor(num_leaves=31, objective='mse')
model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
        eval_metric='mse', early_stopping_rounds=5)
```

## 正式训练
正式训练还是得使用lgb风格的函数来训练，而不是上面的sklearn风格，这样才能调用保存模型的函数。
```
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

num_iterations = 100
#设定param为找到的最佳参数
param = gbm.best_params_
#在此基础上，增加必要的参数
param['num_leaves'] = 31
param['objective'] = 'mse'
param['boosting'] = 'gbrt'
param['metric'] = 'mse'
model = lgb.train(param, train_data, num_iterations, valid_sets=[test_data], early_stopping_rounds=5)

#如果设定了early_stopping_rounds提前结束，就可以这么预测
ypred = model.predict(data, num_iteration=bst.best_iteration)
```

## 模型保存
保存
```
model.save_model('model.txt')
```
导入必须使用Booster
```
import lightgbm as lgb
model = lgb.Booster(model_file='model.txt')
fx = model.predict(data_X)
```

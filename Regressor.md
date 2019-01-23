# 快速确定回归学习器

## 分割数据

```python
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=0) 
#random_state=0 可以每次取样都不一样
```

## 建立管道并验证

```python

choice = 1

if choice==1: 
    #### 线性回归 ####
    from sklearn.linear_model import LinearRegression
    Regressor = LinearRegression(normalize = True)
elif choice==2:
    #### 决策树回归 ####
    from sklearn.tree import DecisionTreeRegressor
    Regressor = DecisionTreeRegressor()
elif choice==3:
    #### SVM回归 ####
    from sklearn.svm import SVR
    Regressor = SVR()
elif choice==4:
    #### KNN回归 ####
    from sklearn.neighbors import KNeighborsRegressor
    Regressor = KNeighborsRegressor()   
elif choice==5:
    #### 随机森林回归 ####
    from sklearn.ensemble import RandomForestRegressor
    Regressor = RandomForestRegressor(n_estimators=50)   
elif choice==6:
    #### Adaboost回归 ####
    from sklearn.ensemble import AdaBoostRegressor
    Regressor = AdaBoostRegressor(n_estimators=50)
elif choice==7:
    #### GBRT回归 ####
    from sklearn.ensemble import GradientBoostingRegressor
    Regressor = GradientBoostingRegressor(n_estimators=100)  
elif choice==8:
    #### Bagging回归 ####
    from sklearn.ensemble import BaggingRegressor
    Regressor = BaggingRegressor(n_estimators=100)  
elif choice==9:
    #### ExtraTree极端随机树回归 ####
    from sklearn.ensemble import ExtraTreeRegressor
    Regressor = ExtraTreeRegressor() 
 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
model = Pipeline(steps=[
    ('sc', StandardScaler()),
    ('rg', Regressor ),
])
#验证
from sklearn.metrics import mean_squared_error   
from sklearn import metrics #保证了scoring可以直接用字符串
scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
scores = 1/(1+ abs(scores))
print(scores)

```

## 训练后保存
注意```X_train```修改为整体训练数据
```python
model.fit(X_train, y_train)
from sklearn.externals import joblib
joblib.dump(model,'rg.model')
```
导入模型
```python
from sklearn.externals import joblib
model=joblib.load('rg.model')
fx = model.predict(data_X)
```

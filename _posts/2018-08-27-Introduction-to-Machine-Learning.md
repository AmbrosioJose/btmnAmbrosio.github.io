---
layout: post
title:  "Introduction to Machine Learning"
description: Machine Learning Basics
img:
date: 2018-08-27  +0200
---
## Day 3, Introduction to Machine Learning

# Introduction to Machine Learning

### Q: In your own words, describe machine learning.

A : Programming the computer so that it can learn on it's own. Train with algorithm and data and a function will be created that can be applied to unseen data aor test data.

### Q: What is your most interested use of machine learning in companies?

A : Robotics

### Q: Can you think of an example where we would use machine learning in your industry?

A : To predict what a website customer might want to buy.

### Q: Can you think of an example of when you would NOT use machine learning in your industry?

A : To predict the amount of users that will view an item.

### Q: What are the two main paths of machine learning?

A : Supervised and unsupervised

### Q: Which is more common, supervised or unsupervised?

A : Supervised

### Q: In your own words, describe supervised learning.

A : Supervised learning is providing supervised data that was provided with appropriate labels and used inpast experiences. Maps x inputs into an output y.

**Challenge**: Name a scenario that requires supervised learning.

A : car loan approvals based on given information and previous experiences.

### Q: In your own words, describe unsupervised learning.

A : Data exists but there wasn't any previous history of the outcome. No label to show acutal results. the algorithm attempts to find structure between data. it looks for natural grouping or clusters. Does not mean a lack of human supervision just an absence of desired or ideal output.

**Challenge**: Name a scenario that requires unsupervised learning.

A : Classifying a large amount of textbooks into categories.

### Q: In your own words, describe semi-supervised learning.

A : Small amount of labeled data with a large amount of unlabeled data. It falls between supervised and unsupervised learning.

**Challenge**: Name a scenario that is best suited for semi-supervised learning.

A : When labeled data is hard to get but unlabeled is readily available. Example: Speech analysis. Data provided by SwitchBoard dataset, desired output: telephone conversation transcripts. It takes 400 hours of annotaion time for each hour of speech.

**Q**: What role does unsupervised learning play in fraud detection?

A : Unsupervised learning can help detect anomolies if transactions deviate too much from their peers which can indicate fraud. Important inputs can be transaction amount and location.

**Q**: What role does supervised learning play in fraud detection?

A : Provides classification rules to detect future cases of fraud.

### Q: In your own words, describe reinforcement learning.

A : Uses different inputs to see the positive or negative outcomes towards a goal

**Challenge**: Name a scenario that is best suited for reinforcement learning.

A :

**Challenge**: Come up with at least five goals a self-driving car is required to meet.

A :<br> Avoid crashing <br>
follow speed limits<br>
Stay within lines when appropriate<br>
Head in the correct direction<br>
Drive safely<br>


## Q: What are the main differences between classification & regression?

A : regression has continous data and classification has categorical data.

### Q: In your own words, describe clustering.

A : grouping a set of object s in such a way that those in the same group are more similar than to those within another group

### Q: In your own words, describe dimension reduction.

A : Reducing the number of random variables and just using a set of principle variables.

# Case Study: HR

### Q: What is our main goal for today's case study?

A :Can we predict what the annual salary is? Which employee s are likely to leave, and which employees should be considered foor a promotion.

Import the data.


```python
import pandas as pd
pd.set_option('display.max_columns', 100)

hr = pd.read_excel('hr.xlsx')

print(hr.info())
hr.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 35 columns):
    Age                         1470 non-null int64
    Attrition                   1205 non-null object
    BusinessTravel              1470 non-null object
    DailyRate                   1470 non-null int64
    Department                  1470 non-null object
    DistanceFromHome            1470 non-null int64
    Education                   1470 non-null int64
    EducationField              1470 non-null object
    EmployeeCount               1470 non-null int64
    EmployeeNumber              1470 non-null int64
    EnvironmentSatisfaction     1470 non-null int64
    Gender                      1470 non-null object
    HourlyRate                  1470 non-null int64
    JobInvolvement              1470 non-null int64
    JobLevel                    1470 non-null int64
    JobRole                     1470 non-null object
    JobSatisfaction             1470 non-null int64
    MaritalStatus               1470 non-null object
    MonthlyIncome               1205 non-null float64
    MonthlyRate                 1470 non-null int64
    NumCompaniesWorked          1470 non-null int64
    Over18                      1470 non-null object
    OverTime                    1470 non-null object
    PercentSalaryHike           1470 non-null int64
    PerformanceRating           1470 non-null int64
    RelationshipSatisfaction    1470 non-null int64
    StandardHours               1470 non-null int64
    StockOptionLevel            1470 non-null int64
    TotalWorkingYears           1470 non-null int64
    TrainingTimesLastYear       1470 non-null int64
    WorkLifeBalance             1470 non-null int64
    YearsAtCompany              1470 non-null int64
    YearsInCurrentRole          1470 non-null int64
    YearsSinceLastPromotion     1470 non-null int64
    YearsWithCurrManager        1470 non-null int64
    dtypes: float64(1), int64(25), object(9)
    memory usage: 402.0+ KB
    None
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobRole</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>MonthlyRate</th>
      <th>NumCompaniesWorked</th>
      <th>Over18</th>
      <th>OverTime</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>Female</td>
      <td>94</td>
      <td>3</td>
      <td>2</td>
      <td>Sales Executive</td>
      <td>4</td>
      <td>Single</td>
      <td>5993.0</td>
      <td>19479</td>
      <td>8</td>
      <td>Y</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>Male</td>
      <td>61</td>
      <td>2</td>
      <td>2</td>
      <td>Research Scientist</td>
      <td>2</td>
      <td>Married</td>
      <td>5130.0</td>
      <td>24907</td>
      <td>1</td>
      <td>Y</td>
      <td>No</td>
      <td>23</td>
      <td>4</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>Male</td>
      <td>92</td>
      <td>2</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Single</td>
      <td>2090.0</td>
      <td>2396</td>
      <td>6</td>
      <td>Y</td>
      <td>Yes</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>Female</td>
      <td>56</td>
      <td>3</td>
      <td>1</td>
      <td>Research Scientist</td>
      <td>3</td>
      <td>Married</td>
      <td>2909.0</td>
      <td>23159</td>
      <td>1</td>
      <td>Y</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>Male</td>
      <td>40</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>2</td>
      <td>Married</td>
      <td>3468.0</td>
      <td>16632</td>
      <td>9</td>
      <td>Y</td>
      <td>No</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Q: What columns do you see?

A :Age, attrition, businessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount and many more.

### Q: Which column(s) is/are closest to helping us achieve our goal?

A : Daily Rate and distance from home.

### 0: Is it true that every Monthly Income is just Daily Rate * 260 / 12?


```python
(hr['DailyRate'] * 260 / 12 == hr['MonthlyIncome']).unique()
```




    array([False])



A : False

### Q: Why do you think this is the case?

A : Some people are salary and others hourly, bonuses and other variables.

## 1: Delete Attrition from our set.


```python
miss = hr.isnull().sum()
miss[miss > 1]

data = hr.copy()
del data['Attrition']
```

## 2: Define your training set.


```python
missing = data[data['MonthlyIncome'].isnull()]

```

### Q: Why should the training set have all values filled?

A : Because algorithms do not take in null values

## 3: Define your implement set.


```python
filled = data[data['MonthlyIncome'].notnull()]
# filled['MonthlyIncome'].max()
# filled['MonthlyIncome'].min()
# filled['MonthlyIncome'].mean()
```

### Q: Why should the implement set have all values missing?

A : The other data is not necessary or can be noise.

# Simple Linear

Import package.


```python
from scipy import stats

```

Fit the data.


```python
# first argument is x second is y
est = stats.linregress(filled['DailyRate'], filled['MonthlyIncome'])
est
```




    LinregressResult(slope=0.11307081909163916, intercept=6432.73027665296, rvalue=0.009596988756751414, pvalue=0.7392828108888272, stderr=0.33967433817855286)



### Q: What form does simple linear regression create as a model?

A : least squared

**Q**: What is your slope?

slope : 0.1130

**Q**: What is your intercept?

6432.73

**Q**: How much can our model be off by?

$1000

## Assess the Model.

Create predictions.


```python
# example of accessing slope
est.slope
# example of accesing intercept
est.intercept

# define predict funtion 
def predict (x, slope, intercept):
    return (slope * x) + intercept
```

### Assess R-squared.


```python
from sklearn.metrics import r2_score, mean_squared_error
r2_score(filled['MonthlyIncome'], (predict( filled['DailyRate'], est.slope, est.intercept)))
```




    9.210219319732982e-05



**Q**: In your own words, describe what R-squared is.

A :It is a statistical measure of how close date is to the regresion line. Values between 0 and 1 or sometimes stated as percentage. It is the percentage of the response variable variation that is explained by a linear model. Generally high value is good and low is bad but not always.

**Q**: Is this a good R-Squared? Why or why not?

A : Not a good R-squared because we want it close to 1.

### Assess RMSE.


```python
mean_squared_error(filled['MonthlyIncome'],predict( filled['DailyRate'], est.slope, est.intercept)) ** .5
```




    4754.022199849941



**Q**: In your own words, describe what RMSE is.

A : Root mean squared error

**Q**: Is this a good RMSE? Why or why not?

A : No because the RMSE value is higher than $1000

## Q: Why does simple linear not work?


```python
import matplotlib.pyplot as plt
%matplotlib inline
filled = hr[hr['DailyRate'].notnull()]

plt.scatter(filled['DailyRate'], filled['MonthlyIncome'])
```




    <matplotlib.collections.PathCollection at 0x16e52ed1278>




![png](output_95_1.png)


A : The data is not linear at all.It is scattered.

## Prep for Multi-variable Modeling


```python
filled.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1470 entries, 0 to 1469
    Data columns (total 35 columns):
    Age                         1470 non-null int64
    Attrition                   1205 non-null object
    BusinessTravel              1470 non-null object
    DailyRate                   1470 non-null int64
    Department                  1470 non-null object
    DistanceFromHome            1470 non-null int64
    Education                   1470 non-null int64
    EducationField              1470 non-null object
    EmployeeCount               1470 non-null int64
    EmployeeNumber              1470 non-null int64
    EnvironmentSatisfaction     1470 non-null int64
    Gender                      1470 non-null object
    HourlyRate                  1470 non-null int64
    JobInvolvement              1470 non-null int64
    JobLevel                    1470 non-null int64
    JobRole                     1470 non-null object
    JobSatisfaction             1470 non-null int64
    MaritalStatus               1470 non-null object
    MonthlyIncome               1205 non-null float64
    MonthlyRate                 1470 non-null int64
    NumCompaniesWorked          1470 non-null int64
    Over18                      1470 non-null object
    OverTime                    1470 non-null object
    PercentSalaryHike           1470 non-null int64
    PerformanceRating           1470 non-null int64
    RelationshipSatisfaction    1470 non-null int64
    StandardHours               1470 non-null int64
    StockOptionLevel            1470 non-null int64
    TotalWorkingYears           1470 non-null int64
    TrainingTimesLastYear       1470 non-null int64
    WorkLifeBalance             1470 non-null int64
    YearsAtCompany              1470 non-null int64
    YearsInCurrentRole          1470 non-null int64
    YearsSinceLastPromotion     1470 non-null int64
    YearsWithCurrManager        1470 non-null int64
    dtypes: float64(1), int64(25), object(9)
    memory usage: 413.4+ KB
    


```python
filled.select_dtypes(exclude='number').info()
# use this to just print numbers can use include or exclude
#filled.select_dtypes(include='number').info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1470 entries, 0 to 1469
    Data columns (total 9 columns):
    Attrition         1205 non-null object
    BusinessTravel    1470 non-null object
    Department        1470 non-null object
    EducationField    1470 non-null object
    Gender            1470 non-null object
    JobRole           1470 non-null object
    MaritalStatus     1470 non-null object
    Over18            1470 non-null object
    OverTime          1470 non-null object
    dtypes: object(9)
    memory usage: 114.8+ KB
    

Get dummies ~


```python
#create dummy variables(make categorical values into separete number values)
X = pd.get_dummies(filled, drop_first=True)
# drop all the values that have a null value
X.dropna(inplace=True)
```

# Multi-variable Regression

Import the algorithm.


```python
from sklearn.linear_model import LinearRegression
```

Set the parameters.


```python
L= LinearRegression()
```

## Q: What form does a model created from multi-variable regression take?

A :

Fit the data.


```python
L.fit(X.drop('MonthlyIncome', axis = 1), X['MonthlyIncome'])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



Assess metrics.


```python
r2_score(X['MonthlyIncome'], L.predict(X.drop('MonthlyIncome', axis = 1)))
mean_squared_error(X['MonthlyIncome'], L.predict(X.drop('MonthlyIncome', axis = 1))) ** .5
```




    1085.5949317240334



# Bias & Variance

### Q: Describe bias in your own words.

A : Measurement of how accurate outputs are over many measurements. Amount of inaccuracy.

### Q: Describe variance in your own words.

A : The meaure of how much data within a group varies. Amount of imprecision,

### Q: What are the key metrics for regression?

A :RMSE( Root Mean Squared Error), R-squared, Adjusted R-squared, MAE(Mean Absolute Error), and RMLSE(Root Mean Log Squared Error)

**Q**: When would you use MAE or RMLSE instead of RMSE?

A :Mae would be used when target is very skewed and MAE when Target is not skewed.

## Q: In this use case, what are our goals?

A : Goals are to predict monthly income? RMSE < $1000

# Ridge

Import algorithm.


```python
from sklearn.linear_model import Ridge

```

Set parameters.


```python
R = Ridge()
```


```python
X.info()

```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1205 entries, 0 to 1469
    Data columns (total 48 columns):
    Age                                  1205 non-null int64
    DailyRate                            1205 non-null int64
    DistanceFromHome                     1205 non-null int64
    Education                            1205 non-null int64
    EmployeeCount                        1205 non-null int64
    EmployeeNumber                       1205 non-null int64
    EnvironmentSatisfaction              1205 non-null int64
    HourlyRate                           1205 non-null int64
    JobInvolvement                       1205 non-null int64
    JobLevel                             1205 non-null int64
    JobSatisfaction                      1205 non-null int64
    MonthlyIncome                        1205 non-null float64
    MonthlyRate                          1205 non-null int64
    NumCompaniesWorked                   1205 non-null int64
    PercentSalaryHike                    1205 non-null int64
    PerformanceRating                    1205 non-null int64
    RelationshipSatisfaction             1205 non-null int64
    StandardHours                        1205 non-null int64
    StockOptionLevel                     1205 non-null int64
    TotalWorkingYears                    1205 non-null int64
    TrainingTimesLastYear                1205 non-null int64
    WorkLifeBalance                      1205 non-null int64
    YearsAtCompany                       1205 non-null int64
    YearsInCurrentRole                   1205 non-null int64
    YearsSinceLastPromotion              1205 non-null int64
    YearsWithCurrManager                 1205 non-null int64
    Attrition_Yes                        1205 non-null uint8
    BusinessTravel_Travel_Frequently     1205 non-null uint8
    BusinessTravel_Travel_Rarely         1205 non-null uint8
    Department_Research & Development    1205 non-null uint8
    Department_Sales                     1205 non-null uint8
    EducationField_Life Sciences         1205 non-null uint8
    EducationField_Marketing             1205 non-null uint8
    EducationField_Medical               1205 non-null uint8
    EducationField_Other                 1205 non-null uint8
    EducationField_Technical Degree      1205 non-null uint8
    Gender_Male                          1205 non-null uint8
    JobRole_Human Resources              1205 non-null uint8
    JobRole_Laboratory Technician        1205 non-null uint8
    JobRole_Manager                      1205 non-null uint8
    JobRole_Manufacturing Director       1205 non-null uint8
    JobRole_Research Director            1205 non-null uint8
    JobRole_Research Scientist           1205 non-null uint8
    JobRole_Sales Executive              1205 non-null uint8
    JobRole_Sales Representative         1205 non-null uint8
    MaritalStatus_Married                1205 non-null uint8
    MaritalStatus_Single                 1205 non-null uint8
    OverTime_Yes                         1205 non-null uint8
    dtypes: float64(1), int64(25), uint8(22)
    memory usage: 280.1 KB
    

Fit the data.


```python
R.fit(X.drop(['MonthlyIncome'], axis = 1), X['MonthlyIncome'])

```




    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001)



Assess metrics.


```python
r2_score(X['MonthlyIncome'], R.predict(X.drop('MonthlyIncome',axis =1)))
mean_squared_error(X['MonthlyIncome'], R.predict(X.drop('MonthlyIncome', axis=1))) ** .5
```




    1086.1610260337507



## Q: What form does ridge cause the model to look like?

A : The ridge caused the model to output a similare value it is just slightly higher.

## Q: Which did better: multi-variate or ridge?

A : Ridge

# Lasso

Import algorithm.


```python
from sklearn.linear_model import Lasso
```

Set parameters.


```python
La = Lasso()
```

Fit the data.


```python
La.fit(X.drop('MonthlyIncome', axis = 1), X['MonthlyIncome'])

```




    Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)



Assess metrics.


```python
r2_score(X['MonthlyIncome'], La.predict(X.drop('MonthlyIncome', axis =1))) ** .5
```




    0.9735533068064492




```python
mean_squared_error(X['MonthlyIncome'], La.predict(X.drop('MonthlyIncome',axis =1))) ** .5
```




    1086.153798390208



### Q: What form does our model take when we use lasso?

A : Low bias and very low variance

### Q: Which model is doing best now?

A : Lasso

# Elastic Net

Import algorithm.


```python
from sklearn.linear_model import ElasticNet
```

Set parameters.


```python
E = ElasticNet()
```

Fit the data.


```python
E.fit(X.drop('MonthlyIncome', axis = 1), X['MonthlyIncome'])
```




    ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
          max_iter=1000, normalize=False, positive=False, precompute=False,
          random_state=None, selection='cyclic', tol=0.0001, warm_start=False)



Assess metrics.


```python
r2_score(X['MonthlyIncome'], E.predict(X.drop('MonthlyIncome', axis =1))) ** .5
```




    0.9183021578078556




```python
mean_squared_error(X['MonthlyIncome'], E.predict(X.drop('MonthlyIncome',axis =1))) ** .5
```




    1882.1100198496429



### Q: What form does our model take when we use elastic-net?

A : High Bias, Good variance

### Q: Which model is doing best now?

A : Ridge

## Q: What are the steps to create a machine learning based model?

A : <br>
1. Define training set<br>
2. Set acceptable rate of error. <br>
3. Chosen several algorithms to create model form<br>
4. Fit the data on all algorithms <br>
5. Printed & assess metrics for all models<br>
6. Decided on the best model based on bias & variance<br>

## Q: What are the steps to use the model you created?

A : <br>
1. Define implement set<br>
2. Configure implement set to same format as training set. <br>
3. Create predictions using the implement set<br>
4. Validate Results <br>
5. Create a DataFrame to store values.<br>

## 1: Define your implement set.


```python
missing = data[data['MonthlyIncome'].isnull()]
filled = data[data['MonthlyIncome'].notnull()]
missing.info()

```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 265 entries, 367 to 1465
    Data columns (total 34 columns):
    Age                         265 non-null int64
    BusinessTravel              265 non-null object
    DailyRate                   265 non-null int64
    Department                  265 non-null object
    DistanceFromHome            265 non-null int64
    Education                   265 non-null int64
    EducationField              265 non-null object
    EmployeeCount               265 non-null int64
    EmployeeNumber              265 non-null int64
    EnvironmentSatisfaction     265 non-null int64
    Gender                      265 non-null object
    HourlyRate                  265 non-null int64
    JobInvolvement              265 non-null int64
    JobLevel                    265 non-null int64
    JobRole                     265 non-null object
    JobSatisfaction             265 non-null int64
    MaritalStatus               265 non-null object
    MonthlyIncome               0 non-null float64
    MonthlyRate                 265 non-null int64
    NumCompaniesWorked          265 non-null int64
    Over18                      265 non-null object
    OverTime                    265 non-null object
    PercentSalaryHike           265 non-null int64
    PerformanceRating           265 non-null int64
    RelationshipSatisfaction    265 non-null int64
    StandardHours               265 non-null int64
    StockOptionLevel            265 non-null int64
    TotalWorkingYears           265 non-null int64
    TrainingTimesLastYear       265 non-null int64
    WorkLifeBalance             265 non-null int64
    YearsAtCompany              265 non-null int64
    YearsInCurrentRole          265 non-null int64
    YearsSinceLastPromotion     265 non-null int64
    YearsWithCurrManager        265 non-null int64
    dtypes: float64(1), int64(25), object(8)
    memory usage: 72.5+ KB
    

## 2: Delete the column of the target variable.


```python
# didn't delete column of target variable. Instead dropped it when calling predict as you will see below when predict is called.
#del missing['MonthlyIncome']
missing.head()
missing.shape
```




    (265, 34)




```python
filled.shape
```




    (1205, 34)



### Q: Why do we have to delete the column?

A : This is our target. We are trying to forecast this column.

## 3: Configure the implement set to be equal to the training set.
For this use case, we just need to create dummies.


```python
predictions = R.predict(missing)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-37-ba53bc992063> in <module>()
    ----> 1 predictions = R.predict(missing)
    

    ~\Anaconda2\envs\Py36\lib\site-packages\sklearn\linear_model\base.py in predict(self, X)
        254             Returns predicted values.
        255         """
    --> 256         return self._decision_function(X)
        257 
        258     _preprocess_data = staticmethod(_preprocess_data)
    

    ~\Anaconda2\envs\Py36\lib\site-packages\sklearn\linear_model\base.py in _decision_function(self, X)
        237         check_is_fitted(self, "coef_")
        238 
    --> 239         X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        240         return safe_sparse_dot(X, self.coef_.T,
        241                                dense_output=True) + self.intercept_
    

    ~\Anaconda2\envs\Py36\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        446         # make sure we actually converted to numeric:
        447         if dtype_numeric and array.dtype.kind == "O":
    --> 448             array = array.astype(np.float64)
        449         if not allow_nd and array.ndim >= 3:
        450             raise ValueError("Found array with dim %d. %s expected <= 2."
    

    ValueError: could not convert string to float: 'No'


## 4: Create & validate predictions.


```python
Y = pd.get_dummies(filled, drop_first = True)
X = pd.get_dummies(missing, drop_first = True)

# now could also delete monthly income from missing but this way is okay as well
# Train the model
R.fit(Y.drop(['MonthlyIncome'], axis = 1), Y['MonthlyIncome'])


```




    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=None, solver='auto', tol=0.001)




```python
# predict based on training
predictions = R.predict(X.drop(['MonthlyIncome'], axis = 1))
predictions
```




    array([ 9582.81962637,  5906.81947046,  2608.52161956,  1961.18641015,
            2757.48455941,  6235.2483267 ,  2442.1946287 ,  5617.66971933,
            9665.09401645,  6081.49533419,  2580.36737595,  5823.3031715 ,
           16592.13297282,  5878.99517001,  2161.42311726,  2780.43197202,
            2186.73913204,  8913.55479987,  2802.30863307,  2867.83745305,
            5669.39143299,  2337.7756757 ,  6337.6120807 , 19222.76246442,
            2499.2811766 ,  5731.30729152,  5306.4685108 ,  2182.02742764,
            6107.58861726,  2922.875527  ,  5451.74806057,  2415.14460151,
            8503.06881415,  5803.45105628,  6203.91070197,  6452.44685643,
            8730.19273002,  6070.28493958,  2718.33812023,  2484.25721137,
            3022.37605991,  2147.10950233,  2489.33855734,  5569.52418085,
            5669.47477786,  2953.26032454,  5871.05221383,  5702.99783652,
            2554.07320816,  2970.40973828,  8802.05906561,  5711.31875704,
            6580.8487388 ,  5802.32474015,  6223.36656449,  5660.1258076 ,
            8526.32758981, 12815.39857025,  6186.36175041,  9085.25009918,
           16582.61874737, 19170.20296369,  5858.70716724,  8486.25674076,
           18745.86718628,  2205.1651272 ,  3121.47859421, 12348.29103332,
            5720.88321536,  5668.07351427,  2533.66365565, 18995.5024531 ,
            9040.41433168,  3012.48307241,  8857.14706696, 16553.53706786,
            9113.89474176,  2312.64008769,  2393.32761982,  6004.18808796,
            3194.58268385,  8738.41440596,  2484.90397312,  5728.23470759,
            2461.73296761,  9097.36099718,  9228.69714214,  2523.31535112,
            9305.30036874,  5432.46095463,  9536.24931482,  8880.9706606 ,
            8869.39533244,  5687.16609717,  2373.84291044,  5843.89284063,
            5811.2129223 ,  2893.22280054,  2211.67888234,  2756.16957433,
            5828.38910245, 11835.38442154,  6045.54538511,  6020.60539108,
            2679.13394669,  2773.22371682, 15816.99058872,  2463.43710459,
            2543.57525928,  8565.17077577,  2812.10187344,  2265.7118969 ,
            8252.03780857,  5903.15441126,  8679.60762447, 16800.21379705,
           19295.7543829 , 16466.98548609,  6338.76915759,  2271.46422178,
            5905.12136248, 15512.60291804,  2743.73504532,  2240.04026968,
            2644.32023022,  5310.93196886,  5958.79126802,  3250.83572537,
            2492.24277691,  2400.10174283,  2513.18629165, 16145.06918022,
            5826.90902911,  5798.37065694,  2748.36166478,  2674.83485112,
            4971.73537799,  8878.55474241,  9491.63943069,  2636.13790306,
            8673.80321008,  9553.10300985,  2943.52031429,  2411.2272744 ,
           19511.04225898,  5688.94067667,  2217.9212403 ,  6164.67385884,
           12021.06347548,  2772.01374316,  5859.3485673 ,  2287.37397878,
            2315.46800635,  2497.39266791,  5998.72398756, 13025.05800366,
            5719.89431023, 19219.08483508,  9315.12215076,  2643.19447406,
            6215.84815535,  6049.46676956,  3249.59066269,  2727.02267595,
           12951.33807026,  5624.44501747,  2493.26720548,  6106.4231576 ,
            5864.26452165,  3161.90484821,  5556.24576195,  5737.81247065,
            5811.32693349,  2366.90575227,  5983.40751531,  9592.09754302,
            8756.35589186,  2844.81867043,  5946.08097835,  6257.16258175,
            5811.74281216, 16585.51437549,  2492.58870645,  9872.48935653,
            6203.00995316,  5533.68469776,  8800.51514146,  2588.30175698,
            6107.126473  ,  5722.00059622, 16232.11731155,  2389.60192925,
            2595.40477854,  3078.16908251,  5599.41057875,  5855.84924695,
            5819.90160031,  2712.53780322,  2647.1468122 ,  5857.97748141,
            2729.80697469,  2475.33454243,  8717.42982804,  2542.26206394,
            6339.30597304,  2727.34154939,  6043.67623018, 12092.9912838 ,
            5289.55941804,  2653.4434537 , 18856.37384521, 19664.53503208,
            2666.45223632,  8746.41238796,  3053.18422661,  5431.59371433,
            3394.65816496,  2348.60666703,  2218.34708501,  2538.26819053,
            6106.55586461,  5347.29759951,  8895.42157058,  2803.27145505,
            5993.83362695,  5427.24707118,  5639.25005361,  6126.85289121,
           16167.35540141,  2674.43368432,  5561.0175919 , 16130.81050843,
            5925.36406125,  2916.46590134,  2423.9733992 ,  6095.09497239,
            5931.96876008, 13329.90794558,  5903.75973564,  6382.32399066,
            5658.2364768 ,  2640.43089087, 18935.29246705,  3015.6207592 ,
           11782.99489688,  5821.32846517,  6264.80619548,  6172.39825259,
            2708.22477122,  8433.90128379,  5763.81982634,  6093.16432319,
            6070.46040501,  5714.54783256,  3046.37769707,  6164.45417839,
            3327.81110451,  2701.66311335,  5616.38566127,  2554.40105225,
            9268.39246645, 12212.50819922,  5708.41661051,  2469.71042276,
            5758.61369442])




```python
predictions.mean()
```




    6314.630493426682




```python
predictions.min()
```




    1961.186410146078




```python
predictions.max()
```




    19664.53503207852



### Q: How would you know these values are valid?

A :  Analyze mean min and max and compare to training data.

## 5: Insert your predictions back into the original dataset.


```python
missing['MonthlyIncome'] = predictions
# Another method but still get a warning
#missing.loc[:, 'MonthlyIncome'] = predictions
```

    C:\Users\joeyr_000\Anaconda2\envs\Py36\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    

## 6: Create a DataFrame to store the results.


```python
results = missing.append(filled).sort_values('EmployeeNumber')
results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>Gender</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobRole</th>
      <th>JobSatisfaction</th>
      <th>MaritalStatus</th>
      <th>MonthlyIncome</th>
      <th>MonthlyRate</th>
      <th>NumCompaniesWorked</th>
      <th>Over18</th>
      <th>OverTime</th>
      <th>PercentSalaryHike</th>
      <th>PerformanceRating</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>Female</td>
      <td>94</td>
      <td>3</td>
      <td>2</td>
      <td>Sales Executive</td>
      <td>4</td>
      <td>Single</td>
      <td>5993.0</td>
      <td>19479</td>
      <td>8</td>
      <td>Y</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>Male</td>
      <td>61</td>
      <td>2</td>
      <td>2</td>
      <td>Research Scientist</td>
      <td>2</td>
      <td>Married</td>
      <td>5130.0</td>
      <td>24907</td>
      <td>1</td>
      <td>Y</td>
      <td>No</td>
      <td>23</td>
      <td>4</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>Male</td>
      <td>92</td>
      <td>2</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>Single</td>
      <td>2090.0</td>
      <td>2396</td>
      <td>6</td>
      <td>Y</td>
      <td>Yes</td>
      <td>15</td>
      <td>3</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>Female</td>
      <td>56</td>
      <td>3</td>
      <td>1</td>
      <td>Research Scientist</td>
      <td>3</td>
      <td>Married</td>
      <td>2909.0</td>
      <td>23159</td>
      <td>1</td>
      <td>Y</td>
      <td>Yes</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>Male</td>
      <td>40</td>
      <td>3</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>2</td>
      <td>Married</td>
      <td>3468.0</td>
      <td>16632</td>
      <td>9</td>
      <td>Y</td>
      <td>No</td>
      <td>12</td>
      <td>3</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# creating a new dataFrame to send just results
result = pd.DataFrame(columns = ['EmployeeNumber','MonthlyIncome'])
result['EmployeeNumber'] = missing['EmployeeNumber']
result['MonthlyIncome'] = predictions
result.head()
# write to csv
results.to_csv('estimate monthly income.csv', index = False)
```

### Q: Why did we pick EmployeeNumber to be the key?

A : Employee number is a much more useful index to company. HR will have an easier time finding employee.

# Polynomial Regressions

## Q: List all of the possible polynomial functions.

A : Power Function Regression, Log/exponential Regressions, Weibull Regression.

### Q: Which one is most applicable to your industry?

A : Power Function.

## Please start a new notebook for the Kings Count House Pricing project.

Thank you for completing your third notebook!

Sign Your Name | Level of Understanding of Regressions
--- | --- 
Jose Ambrosio | 2

& email this notebook to info@thedevmasters.com

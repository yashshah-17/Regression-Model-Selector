

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm

# Importing the dataset
dataset = pd.read_csv('final_output_clean.csv')

# Taking care of missing data
ds = dataset.iloc[:, 1: ].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(ds)
ds = imputer.transform(ds)
X = ds[:, 0:-1]
y = ds[:, -1]
# y = y/1000
row_length = X.shape[0]
column_length = dataset.shape[1]


#########                                   Understanding the Columns                      ##########
X_dash = np.append(arr = np.ones((row_length, 1)).astype(int), values = X, axis = 1)
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_1 = backwardElimination(X_dash, SL)
if X_1[0,0]==1.0:
    X_1 = X_1[:, 1:]
print('Case-1:', len(X_1[0]))
SL = 0.03
X_2 = backwardElimination(X_dash, SL)
if X_2[0,0]==1.0:
    X_2 = X_2[:, 1:]
print('Case-2:', len(X_2[0]))
SL = 0.01
X_3 = backwardElimination(X_dash, SL)
if X_3[0,0]==1.0:
    X_3 = X_3[:, 1:]
print('Case-3:', len(X_3[0]))
SL = 0.005
X_4 = backwardElimination(X_dash, SL)
if X_4[0,0]==1.0:
    X_4 = X_4[:, 1:]
print('Case-4:', len(X_4[0]))
SL = 0.003
X_5 = backwardElimination(X_dash, SL)
if X_5[0,0]==1.0:
    X_5 = X_5[:, 1:]
print('Case-5:', len(X_5[0]))
SL = 0.001
X_6 = backwardElimination(X_dash, SL)
if X_6[0,0]==1.0:
    X_6 = X_6[:, 1:]
print('Case-6:', len(X_6[0])) 
SL = 0.0005
X_7 = backwardElimination(X_dash, SL)
if X_7[0,0]==1.0:
    X_7 = X_7[:, 1:]
print('Case-7:', len(X_7[0]))

#########                                   Multiple Linear Regression                     ##########
def multi_linear(X_opt,y):
    # Splitting 
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # FItting
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    return RMSE



#########                                       SVR Regression                              ##########
def svr(X_opt,y):
    # Splitting 
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test)
    
    # Fitting
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)
    # Predicting a new result
    y_pred = regressor.predict(X_test)
    y_pred = sc_y.inverse_transform(y_pred)
    y_test = sc_y.inverse_transform(y_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    return RMSE

    

#########                                   Decision Tree Regression                        ##########
def decision_tree(X_opt,y):
    # Splitting 
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Fitting 
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    return RMSE



#########                                   Random Forest Regression                        ##########
def random_forest(X_opt,y):
    # Splitting 
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Fitting
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    return RMSE



#########                                      Lasso Regression                             ##########
def lasso(X_opt,y):
    # Splitting
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Fitting
    from sklearn.linear_model import Lasso
    regressor = Lasso(alpha=0.05, normalize=True)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    return RMSE



#########                                   Elastic_net Regression                          ##########
def elastic_net(X_opt,y):
    # Splitting
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Fitting
    from sklearn.linear_model import ElasticNet
    regressor = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    return RMSE



#########                                      Ridge Regression                              ##########
def ridge(X_opt,y):
    # Splitting
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Fitting
    from sklearn.linear_model import Ridge    
    regressor = Ridge(alpha=0.05, normalize=True)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    return RMSE



# Declaring the RMS array
# Array A is for viewing all the RMSEs
# Array B is for computational purpose
A = np.array([['Model', 1, 2, 3, 4, 5, 6, 7], 
              ['Multi-linear', 0, 0, 0, 0, 0, 0, 0],
              ['Decision-tree', 0, 0, 0, 0, 0, 0, 0],
              ['Random-forest', 0, 0, 0, 0, 0, 0, 0],
              ['Lasso', 0, 0, 0, 0, 0, 0, 0],
              ['Elastic_net', 0, 0, 0, 0, 0, 0, 0],
              ['Ridge', 0, 0, 0, 0, 0, 0, 0]])
B = np.array(A[1:7, 1:8], dtype='float')

lowest_value = 99999
    
# Working on different cases
for i in range(1,7):
    for j in range(1,8):
         if j==1:
             X_opt = X_1[:,:]
         elif j==2:
             X_opt = X_2[:,:]
         elif j==3:
             X_opt = X_3[:,:]
         elif j==4:
             X_opt = X_4[:,:]
         elif j==5:
             X_opt = X_5[:,:]
         elif j==6:
             X_opt = X_6[:,:]
         elif j==7:
             X_opt = X_7[:,:]
             
         if i==1:
             A[i,j] = multi_linear(X_opt,y)
             B[i-1,j-1] = A[i,j]
         elif i==2:
             A[i,j] = decision_tree(X_opt,y)
             B[i-1,j-1] = A[i,j]
         elif i==3:
             A[i,j] = random_forest(X_opt,y)
             B[i-1,j-1] = A[i,j]
         elif i==4:
             A[i,j] = lasso(X_opt,y)
             B[i-1,j-1] = A[i,j]
         elif i==5:
             A[i,j] = elastic_net(X_opt,y)
             B[i-1,j-1] = A[i,j]
         elif i==6:
             A[i,j] = ridge(X_opt,y)
             B[i-1,j-1] = A[i,j]
                      
         if (lowest_value > B[i-1,j-1]):
             lowest_value = B[i-1,j-1] 
             i_final = i
             j_final = j




# The final part
if j_final==1:
    X_opt = X_1[:,:]
    c = X_opt.shape[1]
elif j_final==2:
    X_opt = X_2[:,:]
    c = X_opt.shape[1]
elif j_final==3:
    X_opt = X_3[:,:]
    c = X_opt.shape[1]
elif j_final==4:
    X_opt = X_4[:,:]
    c = X_opt.shape[1]
elif j_final==5:
    X_opt = X_5[:,:]
    c = X_opt.shape[1]
elif j_final==6:
    X_opt = X_6[:,:]
    c = X_opt.shape[1]
elif j_final==7:
    X_opt = X_7[:,:]
    c = X_opt.shape[1]
             
    
if i_final==1:
    # Splitting 
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # FItting
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    print('Multi-linear Regression with %i columns!' % c)
    print('RMSE = %f' % RMSE)
elif i_final==2:
    # Splitting 
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Fitting 
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    print('Decision-tree Regression with %i columns!' % c)
    print('RMSE = %f' % RMSE)
elif i_final==3:
    # Splitting 
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Fitting
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    print('Random-forest Regression with %i columns!' % c)
    print('RMSE = %f' % RMSE)
elif i_final==4:
    # Splitting
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Fitting
    from sklearn.linear_model import Lasso
    regressor = Lasso(alpha=0.05, normalize=True)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    print('Lasso Regression with %i columns!' % c)
    print('RMSE = %f' % RMSE)
elif i_final==5:
    # Splitting
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Fitting
    from sklearn.linear_model import ElasticNet
    regressor = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    print('Elastic-net Regression with %i columns!' % c)
    print('RMSE = %f' % RMSE)
elif i_final==6:
   # Splitting
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Fitting
    from sklearn.linear_model import Ridge    
    regressor = Ridge(alpha=0.05, normalize=True)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    
    y_pred = regressor.predict(X_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/(y_pred.shape[0])
    RMSE = MSE**(0.5)
    print('Ridge Regression with %i columns!' % c)
    print('RMSE = %f' % RMSE)
elif i_final==7:
    # Splitting 
    y = y.reshape(419,1)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test)
    y = np.array(y).tolist()
    # Fitting
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)
    # Predicting a new result
    y_pred = regressor.predict(X_test)
    y_pred = sc_y.inverse_transform(y_pred)
    y_test = sc_y.inverse_transform(y_test)
    # Calculating MSE and RMSE
    y_final = (y_test-y_pred)*(y_test-y_pred)
    y_sum = sum(y_final)
    MSE = y_sum/105
    RMSE = MSE**(0.5)
    print('SVR Regression with %i columns!' % c)
    print('RMSE = %f' % RMSE)

     
# Visualising the Training set results
plt.plot(list(range(0,y_pred.shape[0])), y_test, color = 'red')
plt.plot(list(range(0,y_pred.shape[0])), y_pred, color = 'blue')
plt.xlabel('Input')
plt.ylabel('Y-values')
plt.grid()
plt.show()



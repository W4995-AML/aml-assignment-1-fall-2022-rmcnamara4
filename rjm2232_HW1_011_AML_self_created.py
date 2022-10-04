import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import inv
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

pd.options.mode.chained_assignment = None

# load data 
credit_data = pd.read_csv('dataset_credit.csv')

# create subplots and adjust horizontal space between subplots 
fig, ax = plt.subplots(ncols = 1, nrows = 3, figsize = (15, 15))
fig.subplots_adjust(hspace = 0.5)

# plot credit_amount histogram 
sns.histplot(credit_data['credit_amount'], ax = ax[0])

# set x label, y label, and title for credit_amount histogram
ax[0].set_xlabel('Credit Amount ($)')
ax[0].set_ylabel('Frequency') 
ax[0].set_title('Distribution of Credit Amount')

# plot age histogram 
sns.histplot(credit_data['age'], ax = ax[1])

# set x label, y label, and title for age histogram 
ax[1].set_xlabel('Age (years)') 
ax[1].set_ylabel('Frequency') 
ax[1].set_title('Distribution of Age')

# plot duration histogram 
sns.histplot(credit_data['duration'], ax = ax[2])

# set x label, y label, and title for duration histogram
ax[2].set_xlabel('Duration')
ax[2].set_ylabel('Frequency')
ax[2].set_title('Distribution of Duration')

# show figure
plt.show()

# create figure 
fig = plt.figure(figsize = (15, 10))

# get current axes 
ax = fig.gca()

# plot boxplot
sns.boxplot(data = credit_data, x = 'class', y = 'credit_amount', palette = ['green', 'red'])

# set x label, y label, and title
plt.xlabel('Class')
plt.ylabel('Credit Amount ($)')
plt.title('Relationship between Credit Amount and Class')

# change labels of the x tick marks 
plt.xticks(ticks = [0, 1], labels = ['Good', 'Bad'])

# add commas in the numbers on the y axis 
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

# show figure 
plt.show()

# create figure 
fig = plt.figure(figsize = (15, 10))

# get value counts for the class column
counts = credit_data['class'].value_counts()

# create dictionary mapping for the labels
labels = {
    'Good': 'good', 
    'Bad': 'bad'
}

# create dictionary mapping for the colors 
colors = {
    'green': 'good', 
    'red': 'bad'
}

# plot pie chart
plt.pie(counts, labels = labels, colors = colors, autopct = "%.0f%%")
plt.legend()

# show figure 
plt.show()

np.random.seed(0)
epsilon = np.random.normal(0, 3, 100)
x = np.linspace(0, 10, 100) 
# y = np.linspace(0, 5, 100)
y = 5 * x + 10 + epsilon

# create figure 
fig = plt.figure(figsize = (15, 10))

# plot scatter plot
plt.scatter(x, y, label = f'Correlation = {np.round(np.corrcoef(x, y)[0,1], 2)}')

# change x and y ticks 
plt.yticks(ticks = range(0, 80, 10))
plt.xticks(ticks = range(0, 12, 2))

# set x label, y label, and title
plt.xlabel('Synthetic X')
plt.ylabel('Synthetic Y')
plt.title('Scatter plot of Synthetic Y vs Synthetic X')

# add legend to upper left 
plt.legend(loc = 'upper left')

# show figure
plt.show()

# Load auto MPG dataset
auto_mpg_df = pd.read_csv('auto-mpg.csv')

# drop some rows with missing entries
auto_mpg_df = auto_mpg_df[auto_mpg_df['horsepower'] != '?']

# Cast horsepower column to float
auto_mpg_df['horsepower'] = auto_mpg_df['horsepower'].astype(float)

auto_mpg_df

# Split data into features and labels
auto_mpg_X = auto_mpg_df.drop(columns=['mpg'])
auto_mpg_y = auto_mpg_df['mpg']

# create subplots and adjust horizontal space between subplots 
fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (15, 10))
fig.subplots_adjust(hspace = 0.5)

# plot scatter plot of MPG vs displacement 
ax[0, 0].scatter(auto_mpg_X['displacement'], auto_mpg_y)

# set x label, y label, and title of MPG vs displacement 
ax[0, 0].set_xlabel('Displacement')
ax[0, 0].set_ylabel('MPG')
ax[0, 0].set_title('MPG vs Displacement')

# set y ticks 
ax[0, 0].set_yticks(ticks = range(0, 55, 5))

# plot scatter plot of MPG vs horsepower
ax[0, 1].scatter(auto_mpg_X['horsepower'], auto_mpg_y)

# set x label, y label, and title of MPG vs horsepower
ax[0, 1].set_xlabel('Horsepower')
ax[0, 1].set_ylabel('MPG')
ax[0, 1].set_title('MPG vs Horsepower')

# set y ticks
ax[0, 1].set_yticks(ticks = range(0, 55, 5))

# plot scatter plot of MPG vs weight
ax[1, 0].scatter(auto_mpg_X['weight'], auto_mpg_y)

# set x label, y label, and title of MPG vs weight
ax[1, 0].set_xlabel('Weight')
ax[1, 0].set_ylabel('MPG')
ax[1, 0].set_title('MPG vs Weight')

# set y ticks
ax[1, 0].set_yticks(ticks = range(0, 55, 5))

# plot scatter plot of MPG vs acceleration 
ax[1, 1].scatter(auto_mpg_X['acceleration'], auto_mpg_y)

# set x label, y label, and title of MPG vs acceleration 
ax[1, 1].set_xlabel('Acceleration')
ax[1, 1].set_ylabel('MPG')
ax[1, 1].set_title('MPG vs Acceleration')

# set y ticks
ax[1, 1].set_yticks(ticks = range(0, 55, 5))

# show figure 
plt.show()

# create subplots and adjust the horizontal space between subplots 
fig, ax = plt.subplots(ncols = 1, nrows = 3, figsize = (15, 15))
fig.subplots_adjust(hspace = 0.5)

# plot boxplot of MPG vs cylinders 
sns.boxplot(x = auto_mpg_X['cylinders'], y = auto_mpg_y, ax = ax[0])

# set x label, y label, and title of MPG vs cylinders 
ax[0].set_xlabel('Cylinders')
ax[0].set_ylabel('MPG')
ax[0].set_title('MPG vs Cylinders')

# plot boxplot of MPG vs Model Year 
sns.boxplot(x = auto_mpg_X['model year'], y = auto_mpg_y, ax = ax[1])

# set x label, y label, and title of MPG vs Model Year
ax[1].set_xlabel('Model Year')
ax[1].set_ylabel('MPG')
ax[1].set_title('MPG vs Model Year')

# plot boxplot of MPG vs origin
sns.boxplot(x = auto_mpg_X['origin'], y = auto_mpg_y, ax = ax[2])

# set x label, y label, and title of MPG vs origin 
ax[2].set_xlabel('Origin') 
ax[2].set_ylabel('MPG')
ax[2].set_title('MPG vs Origin')

# loop over all of the boxplot outline and make them black 
for i in range(0, 3): 
    for j, box in enumerate(ax[i].artists):
        box.set_edgecolor('black')
        box.set_facecolor('white')
    
        for k in range(6 * j,6 * (j + 1)):
            ax[i].lines[k].set_color('black')

X = x.reshape((100, 1))   # Turn the x vector into a feature matrix X

# 1. No categorical features in the synthetic dataset (skip this step)

# 2. Split the dataset into training (60%), validation (20%), and test (20%) sets
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size = 0.25, random_state = 0)

# 3. Standardize the columns in the feature matrices
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Fit and transform scalar on X_train
X_val = scaler.transform(X_val)           # Transform X_val
X_test = scaler.transform(X_test)         # Transform X_test

# 4. Add a column of ones to the feature matrices
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

print(X_train[:5], '\n\n', y_train[:5])

# split dataset into training, validation, and test sets 
auto_mpg_X_dev, auto_mpg_X_test, auto_mpg_y_dev, auto_mpg_y_test = train_test_split(auto_mpg_X, auto_mpg_y, test_size = 0.2, random_state = 0)
auto_mpg_X_train, auto_mpg_X_val, auto_mpg_y_train, auto_mpg_y_val = train_test_split(auto_mpg_X_dev, auto_mpg_y_dev, test_size = 0.25, random_state = 0)

# standardize the columns in the feature matrices 
scaler = StandardScaler() 
auto_mpg_X_train = scaler.fit_transform(auto_mpg_X_train)
auto_mpg_X_val = scaler.transform(auto_mpg_X_val)
auto_mpg_X_test = scaler.transform(auto_mpg_X_test)

# add a column of ones to the feature matrices 
auto_mpg_X_train = np.hstack([np.ones((auto_mpg_X_train.shape[0], 1)), auto_mpg_X_train])
auto_mpg_X_val = np.hstack([np.ones((auto_mpg_X_val.shape[0], 1)), auto_mpg_X_val])
auto_mpg_X_test = np.hstack([np.ones((auto_mpg_X_test.shape[0], 1)), auto_mpg_X_test])

class LinearRegression():
    '''
    Linear regression model with L2-regularization (i.e. ridge regression).

    Attributes
    ----------
    alpha: regularization parameter
    w: (n x 1) weight vector
    '''
    
    def __init__(self, alpha=0):
        self.alpha = alpha
        self.w = None

    def train(self, X, y):
        '''Trains model using ridge regression closed-form solution 
        (sets w to its optimal value).
        
        Parameters
        ----------
        X : (m x n) feature matrix
        y: (m x 1) label vector
        
        Returns
        -------
        None
        '''
        self.w = np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(self.alpha, np.eye(X.shape[1]))), np.dot(X.T, y))
        
    def predict(self, X):
        '''Predicts on X using trained model.
        
        Parameters
        ----------
        X : (m x n) feature matrix
        
        Returns
        -------
        y_pred: (m x 1) prediction vector
        '''
        y_pred = np.dot(X, self.w)
        return y_pred
        
# create linear regression model with alpha = 0
lin_reg = LinearRegression(alpha = 0)

# train the model on X_train and y_train
lin_reg.train(X_train, y_train)

# predict the y value for X_test 
y_pred = lin_reg.predict(X_test)

# define a list of indexes we want to extract 
indexes = [0, 1, 2, -3, -2, -1]

# print out the predictions and actual labels 
print('Predictions:', y_pred[indexes])
print('Actual Labels:', y_test[indexes])

# plot scatter plot of y_test vs X_test 
plt.scatter(x = X_test[:, 1], y = y_test)

# use the weights of the model to get the predicted y values 
y_check = X_test[:, 1] * lin_reg.w[1] + lin_reg.w[0]

# plot the predicted y values vs the X values 
plt.plot(X_test[:, 1], y_check)

# set the x label and y label 
plt.xlabel('X_test')
plt.ylabel('y_test')

from sklearn.metrics import mean_squared_error

# create linear regression model with alpha = 0
lin_reg = LinearRegression(alpha = 0)

# train the model on auto_mpg_X_train and auto_mpg_y_train
lin_reg.train(auto_mpg_X_train, auto_mpg_y_train)

# use the model to predict the y values for the training, validation, and test sets 
auto_mpg_y_pred_train = lin_reg.predict(auto_mpg_X_train)
auto_mpg_y_pred_val = lin_reg.predict(auto_mpg_X_val)
auto_mpg_y_pred_test = lin_reg.predict(auto_mpg_X_test)

# define function to calculate the mean-squared error 
def mse(y_true, y_pred): 
    return np.mean((y_true - y_pred) ** 2)

# print the mean-squared error on the training, validation, and test sets 
print('The mean-squared error on auto_mpg_y_train is:', mse(auto_mpg_y_train, auto_mpg_y_pred_train))
print('The mean-squared error on auto_mpg_y_val is:', mse(auto_mpg_y_val, auto_mpg_y_pred_val))
print('The mean-squared error on auto_mpg_y_test is:', mse(auto_mpg_y_test, auto_mpg_y_pred_test))

print()

# define a list of indexes we want to extract 
indexes = [0, 1, 2, -3, -2, -1]

# print the predictions on the test set and the actual labels of the test set 
print('Predictions on test set:', auto_mpg_y_pred_test[indexes])
print('Actual labels of test set:', np.array(auto_mpg_y_test)[indexes])

# calculate the mean of the auto_mpg_y_train set 
auto_mpg_y_train_mean = np.mean(auto_mpg_y_train)

# calculate the baseline mean-squared error on the training, validation, and test sets 
mse_mpg_y_train = mse(auto_mpg_y_train, [auto_mpg_y_train_mean] * len(auto_mpg_y_train))
mse_mpg_y_val = mse(auto_mpg_y_val, [auto_mpg_y_train_mean] * len(auto_mpg_y_val))
mse_mpg_y_test = mse(auto_mpg_y_test, [auto_mpg_y_train_mean] * len(auto_mpg_y_test))

# print the mean-squared error on the training, validation, and test sets
print('The baseline mean-squared error on auto_mpg_y_train is:', mse_mpg_y_train)
print('The baseline mean-squared error on auto_mpg_y_val is:', mse_mpg_y_val)
print('The baseline mean-squared error on auto_mpg_y_test is:', mse_mpg_y_test)

# create figure 
fig = plt.figure(figsize = (15, 10))

# get the feature names of auto_mpg_X and add 'bias' to the beginning of the list 
col_names = list(auto_mpg_X.columns)
col_names.insert(0, 'bias')

# plot bar plot of the weights vs the features 
plt.bar(x = range(0, len(lin_reg.w)), height = lin_reg.w)

# set the x and y labels 
plt.xlabel('Feature')
plt.ylabel('Feature Weight')

# set the x ticks as the feature names 
plt.xticks(ticks = range(0, len(lin_reg.w)), labels = col_names)

# show the figure 
plt.show()

# define alphas we want to search over 
alphas = np.logspace(-5, 1, 20)

# initialize empty lists for the train and validation mean-squared errors 
train_mse = []
val_mse = []

# for each alpha, train a model and record the mean-squared error on the training and validation sets 
for idx in alphas: 
    model = LinearRegression(alpha = idx)
    model.train(auto_mpg_X_train, auto_mpg_y_train)
    
    auto_mpg_y_pred_train = model.predict(auto_mpg_X_train)
    train_mse.append(mse(auto_mpg_y_train, auto_mpg_y_pred_train))
    
    auto_mpg_y_pred_val = model.predict(auto_mpg_X_val)
    val_mse.append(mse(auto_mpg_y_val, auto_mpg_y_pred_val))
    
# create figure 
fig = plt.figure(figsize = (15, 10))

# plot the mean-squared error vs alpha for the training and validation sets 
plt.plot(alphas, train_mse, 'b.--', label = 'Train Set')
plt.plot(alphas, val_mse, 'r.--', label = 'Validation Set')

# add legend
plt.legend()

# set the x and y labels 
plt.xlabel('Alpha')
plt.ylabel('Mean-Squared Error')

# change x scale to log
plt.xscale('log')

nba_reg = pd.read_csv("nba_logreg.csv")
nba_reg.head()
nba_reg.shape

# set options to view the whole dataset 
pd.set_option('display.max_rows', 2000)
# print(nba_reg)

# get the counts of how many missing values are in each column 
missing_vals = np.sum(nba_reg.isnull())
print(missing_vals)

# remove the missing values from the '3P%' column
nba_reg_new = nba_reg[~nba_reg['3P%'].isnull()]

# get the percentage of each label
y_probs = nba_reg_new['TARGET_5Yrs'].value_counts() / nba_reg_new.shape[0]
print(y_probs)

nba_X = nba_reg_new.drop(columns=['TARGET_5Yrs'])
nba_y = nba_reg_new['TARGET_5Yrs']
print(nba_X.shape)

# create figure 
fig = plt.figure(figsize = (15, 10))

# get the correlation matrix of nba_X
correlations = nba_X.corr()

# create figure 
fig = plt.figure(figsize = (15, 10))

# get the correlation matrix of nba_X
correlations = nba_X.corr()

# plot heatmap 
sns.heatmap(correlations, cmap = 'RdBu_r', center = 0, annot = True, square = True, cbar_kws = {'location': 'left', 'label': 'Correlation'})

# set x and y label 
plt.xlabel('Feature')
plt.ylabel('Feature')

# show figure 
plt.show()

# get absolute values of the correlations
correlations_abs = nba_X.corr().abs()

# extract the upper triangle 
upper_tri = correlations_abs.where(np.triu(np.ones(correlations_abs.shape),k=1).astype(bool))

# get the features that are highly correlated (>= 0.9)
drop_cols = [col for col in upper_tri.columns if any(upper_tri[col] >= 0.9)]

# drop the features 
nba_X_dropped_cols = nba_X.drop(drop_cols, axis = 1)

# Split data into features and labels
nba_new_X = nba_X_dropped_cols.drop(columns = 'Name', axis = 1)
nba_new_Y = nba_y
print(nba_new_X.columns)

# convert series to a numpy array 
nba_new_Y = nba_new_Y.to_numpy().reshape((nba_new_Y.shape[0], 1))

# split dataset into training, validation, and test sets 
nba_new_X_dev, nba_new_X_test, nba_new_Y_dev, nba_new_Y_test = train_test_split(nba_new_X, nba_new_Y, test_size = 0.2, random_state = 0)
nba_new_X_train, nba_new_X_val, nba_new_Y_train, nba_new_Y_val = train_test_split(nba_new_X_dev, nba_new_Y_dev, test_size = 0.25, random_state = 0)

# standardize the columns of the feature matrix 
scaler = StandardScaler()
nba_new_X_train = scaler.fit_transform(nba_new_X_train)
nba_new_X_val = scaler.transform(nba_new_X_val)
nba_new_X_test = scaler.transform(nba_new_X_test)

# add column of ones to the feature matrices of train, validation, and test datasets 
nba_new_X_train = np.hstack([np.ones((nba_new_X_train.shape[0], 1)), nba_new_X_train])
nba_new_X_val = np.hstack([np.ones((nba_new_X_val.shape[0], 1)), nba_new_X_val])
nba_new_X_test = np.hstack([np.ones((nba_new_X_test.shape[0], 1)), nba_new_X_test])

class LogisticRegression():
    '''
    Logistic regression model with L2 regularization.

    Attributes
    ----------
    alpha: regularization parameter
    t: number of epochs to run gradient descent
    eta: learning rate for gradient descent
    w: (n x 1) weight vector
    '''
    
    def __init__(self, alpha=0, t=100, eta=1e-3):
        self.alpha = alpha
        self.t = t
        self.eta = eta
        self.w = None

    def train(self, X, y):
        '''Trains logistic regression model using gradient descent 
        (sets w to its optimal value).
        
        Parameters
        ----------
        X : (m x n) feature matrix
        y: (m x 1) label vector
        
        Returns
        -------
        losses: (t x 1) vector of losses at each epoch of gradient descent
        '''
        
        loss = list()
        self.w = np.zeros((X.shape[1],1))
        for i in range(self.t): 
            self.w = self.w - (self.eta * self.calculate_gradient(X, y))
            loss.append(self.calculate_loss(X, y))
        return loss
        
    def predict(self, X):
        '''Predicts on X using trained model. Make sure to threshold 
        the predicted probability to return a 0 or 1 prediction.
        
        Parameters
        ----------
        X : (m x n) feature matrix
        
        Returns
        -------
        y_pred: (m x 1) 0/1 prediction vector
        '''
        y_pred = self.calculate_sigmoid(X.dot(self.w))
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred
    
    def calculate_loss(self, X, y):
        '''Calculates the logistic regression loss using X, y, w, 
        and alpha. Useful as a helper function for train().
        
        Parameters
        ----------
        X : (m x n) feature matrix
        y: (m x 1) label vector
        
        Returns
        -------
        loss: (scalar) logistic regression loss
        '''
        return -y.T.dot(np.log(self.calculate_sigmoid(X.dot(self.w)))) - (1-y).T.dot(np.log(1-self.calculate_sigmoid(X.dot(self.w)))) + self.alpha*np.linalg.norm(self.w, ord=2)**2
    
    def calculate_gradient(self, X, y):
        '''Calculates the gradient of the logistic regression loss 
        using X, y, w, and alpha. Useful as a helper function 
        for train().
        
        Parameters
        ----------
        X : (m x n) feature matrix
        y: (m x 1) label vector
        
        Returns
        -------
        gradient: (n x 1) gradient vector for logistic regression loss
        '''
        return X.T.dot(self.calculate_sigmoid( X.dot(self.w)) - y) + 2*self.alpha*self.w        
            
    
    def calculate_sigmoid(self, x):
        '''Calculates the sigmoid function on each element in vector x. 
        Useful as a helper function for predict(), calculate_loss(), 
        and calculate_gradient().
        
        Parameters
        ----------
        x: (m x 1) vector
        
        Returns
        -------
        sigmoid_x: (m x 1) vector of sigmoid on each element in x
        '''
        return (1)/(1 + np.exp(-x.astype('float')))

# create logistic regression model with alpha = 0, t = 100, and eta = 1e-3
log_reg = LogisticRegression(alpha = 0, t = 100, eta = 1e-3)

# train the model and save the loss values 
loss_vals = log_reg.train(nba_new_X_train, nba_new_Y_train)

# squeeze the loss values 
for idx in range(len(loss_vals)): 
    loss_vals[idx] = loss_vals[idx].squeeze()
    
# create figure 
fig = plt.figure(figsize = (15, 10))

# plot loss values vs epoch
plt.plot(range(1, 101), loss_vals)

# set x and y label and the title 
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch for Logistic Regression Training')

# show the figure 
plt.show()

# set seed 
np.random.seed(76)

# initialize lists for the hyperparameters and accuracies 
alphas_list = []
etas_list = []
ts_list = []
accuracy_list = []

# get random values for the hyperparameters and append them to the lists 
# train the model with the hyperparameters 
# predict and record the accuracy 

# I wasn't sure if we were allowed to import other functions for this purpose, however, I think it is helpful to do it 
# myself to understand how it is working
for i in range(20): 
    alpha = np.random.random(1)
    eta = np.random.random(1) / 1000
    t = np.random.randint(0, 101)
    
    alphas_list.append(alpha)
    etas_list.append(eta)
    ts_list.append(t)
    
    log_reg = LogisticRegression(alpha = alpha, eta = eta, t = t)
    
    loss = log_reg.train(nba_new_X_train, nba_new_Y_train)
    
    nba_new_Y_val_pred = log_reg.predict(nba_new_X_val)
    
    accuracy_list.append(accuracy_score(nba_new_Y_val, nba_new_Y_val_pred))

# get the index of the largest accuracy score 
ind = np.argmax(accuracy_list)

# print the max accuracy and the best hyperparameters 
print('The max accuracy on the validation set is:', accuracy_list[ind])
print('This is achieved with the following hyperparamters:')
print('\t- alpha =', np.squeeze(alphas_list[ind]))
print('\t- t =', ts_list[ind])
print('\t- eta = ', np.squeeze(etas_list[ind]))

# I wasn't sure if we were supposed to retrain the model on all of the dev set and then test on the test set, but that 
# is what I did. Technically, as the professor said, this would be the way to actual test your model. 

# resplit the dataset into dev and test sets 
nba_new_X_dev, nba_new_X_test, nba_new_Y_dev, nba_new_Y_test = train_test_split(nba_new_X, nba_new_Y, test_size = 0.2, random_state = 0)
# scale the features 
scaler = StandardScaler()
nba_new_X_dev = scaler.fit_transform(nba_new_X_dev)
nba_new_X_test = scaler.transform(nba_new_X_test)

# add a column of ones to the feature matrices 
nba_new_X_dev = np.hstack([np.ones((nba_new_X_dev.shape[0], 1)), nba_new_X_dev])
nba_new_X_test = np.hstack([np.ones((nba_new_X_test.shape[0], 1)), nba_new_X_test])

# define two logistic regression models with the hyperparameters above 
log_reg_1 = LogisticRegression(alpha = 0, t = 100, eta = 1e-3)
log_reg_2 = LogisticRegression(alpha = np.squeeze(alphas_list[ind]), t = ts_list[ind], eta = np.squeeze(etas_list[ind]))

# train the two models 
loss_1 = log_reg_1.train(nba_new_X_dev, nba_new_Y_dev)
loss_2 = log_reg_2.train(nba_new_X_dev, nba_new_Y_dev)

# predict using the two models 
nba_new_Y_test_pred_1 = log_reg_1.predict(nba_new_X_test)
nba_new_Y_test_pred_2 = log_reg_2.predict(nba_new_X_test)

# calculate the accuracy scores 
nba_new_Y_test_acc_1 = accuracy_score(nba_new_Y_test, nba_new_Y_test_pred_1)
nba_new_Y_test_acc_2 = accuracy_score(nba_new_Y_test, nba_new_Y_test_pred_2)

# print the accuracy scores 
print()
print('The accuracy on the test set with alpha = 0, t = 100, and eta = 1e-3 is:', nba_new_Y_test_acc_1)
print('The accuracy on the test set with alpha =', np.squeeze(alphas_list[ind]), ',', 't =', 
     ts_list[ind], ',', 'and eta =', np.squeeze(etas_list[ind]), 'is:', nba_new_Y_test_acc_2)

# create figure 
fig = plt.figure(figsize = (15, 10))

# get feature names and add 'bias' to the beginning of the list 
col_names = list(nba_new_X.columns)
col_names.insert(0, 'bias')

# plot bar plot of weights vs features 
plt.bar(x = range(0, len(log_reg_2.w)), height = np.squeeze(log_reg_2.w))

# set x and y label 
plt.xlabel('Feature')
plt.ylabel('Feature Weight')

# change x ticks to the feature names 
plt.xticks(ticks = range(0, len(log_reg_2.w)), labels = col_names)

# show figure 
plt.show()

cancer_df = pd.read_csv('breast-cancer.csv')
cancer_df = cancer_df.drop(columns=['id', 'Unnamed: 32'])
cancer_df.head()

# Split data into features and labels
cancer_X = cancer_df.drop(columns=['diagnosis'])
cancer_y = cancer_df['diagnosis']

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

cancer_df['diagnosis'] = cancer_df.diagnosis.astype("category").cat.codes
cancer_y_enc = cancer_df['diagnosis'].to_numpy()
cancer_y_enc = cancer_y_enc.reshape(cancer_y_enc.shape[0],1)
print(cancer_y_enc.shape)
print(type(cancer_y_enc))


cancer_X_dev, cancer_X_test, cancer_y_dev, cancer_y_test = train_test_split(cancer_X, cancer_y_enc, test_size=0.2, random_state=0)
cancer_X_train, cancer_X_val, cancer_y_train, cancer_y_val = train_test_split(cancer_X_dev, cancer_y_dev, test_size=0.25, random_state=0)


scaler = StandardScaler()
cancer_X_train = scaler.fit_transform(cancer_X_train) 
cancer_X_val = scaler.transform(cancer_X_val)
cancer_X_test = scaler.transform(cancer_X_test)


cancer_X_train = np.hstack([np.ones((cancer_X_train.shape[0], 1)), cancer_X_train])
cancer_X_val = np.hstack([np.ones((cancer_X_val.shape[0], 1)), cancer_X_val])
cancer_X_test = np.hstack([np.ones((cancer_X_test.shape[0], 1)), cancer_X_test])

# create primal SVM model
primal_svm = LinearSVC()

# fit the model to the training set 
primal_svm.fit(cancer_X_train, cancer_y_train.reshape(cancer_y_train.shape[0], ))

# predict on the training, validation, and test sets 
cancer_y_pred_train_p = primal_svm.predict(cancer_X_train)
cancer_y_pred_val_p = primal_svm.predict(cancer_X_val)
cancer_y_pred_test_p = primal_svm.predict(cancer_X_test)

# calculate accuracy score on the training, validation, and test sets 
cancer_train_acc_p = accuracy_score(cancer_y_train, cancer_y_pred_train_p)
cancer_val_acc_p = accuracy_score(cancer_y_val, cancer_y_pred_val_p)
cancer_test_acc_p = accuracy_score(cancer_y_test, cancer_y_pred_test_p)

# print the accuracy scores 
print('For primal SVM:')
print('\t- The accuracy on the training set is:', cancer_train_acc_p)
print('\t- The accuracy on the validation set is:', cancer_val_acc_p)
print('\t- The accuracy on the test set is:', cancer_test_acc_p)

# create dual SVM model 
dual_svm = SVC()

# fit the model to the training data 
dual_svm.fit(cancer_X_train, cancer_y_train.reshape(cancer_y_train.shape[0], ))

# predict on the training, validation, and test sets 
cancer_y_pred_train_d = dual_svm.predict(cancer_X_train)
cancer_y_pred_val_d = dual_svm.predict(cancer_X_val)
cancer_y_pred_test_d = dual_svm.predict(cancer_X_test)

# calculate accuracy score on the training, validation, and test sets 
cancer_train_acc_d = accuracy_score(cancer_y_train, cancer_y_pred_train_d)
cancer_val_acc_d = accuracy_score(cancer_y_val, cancer_y_pred_val_d)
cancer_test_acc_d = accuracy_score(cancer_y_test, cancer_y_pred_test_d)

# print accuracy scores 
print('For dual SVM:')
print('\t- The accuracy on the training set is:', cancer_train_acc_d)
print('\t- The accuracy on the validation set is:', cancer_val_acc_d)
print('\t- The accuracy on the test set is:', cancer_test_acc_d)

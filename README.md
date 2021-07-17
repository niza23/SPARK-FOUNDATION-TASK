# By - Nidhi Vijaybhai Zala
# The Sparks Foundation : Data Science and Buisness Analytics (GRIP JULY'21)


# Task 1 - Predicting using Supervised Model
# Predict the percentage of an student based on the no. of study hours

#import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Reading data from the link
s_data = pd.read_csv("http://bit.ly/w-data")
print("Data imported successfully")
s_data.head(10)


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

# Preparing the data
X = s_data.iloc[:, :-1].values
y= s_data.iloc[:, 1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#Training the Algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

print("Training Complete.")


line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# Making Predictions
print(X_test)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


hours = [9.25]
y_pred = regressor.predict([hours])

print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(y_pred[0]))

#Evaluating the model
from sklearn import metrics 
y_pred = regressor.predict(X_test)
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


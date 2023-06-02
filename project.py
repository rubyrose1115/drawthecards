#1,2,4,4,3,5,5   240=70%=168
#IMPORTS
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

#DATA HANDLING
data = pd.read_csv('project.csv')
print(data.head())
print(data.describe())
X = data[["water resource"]]
Y = data["stable"]
Y_ = data["selected"]

#DATA ANALYSIS
plt.scatter(X['water resource'], Y, color='b')
plt.xlabel('water resource')  
plt.ylabel('stable') 
plt.show()

#OBSERVATIONS
print("From the plot we can say that pobability of building a house is more if the water facilities are above 6")

print("-------------------------------------")
#DECISION TREE CLASSIFICATION
from sklearn.tree import DecisionTreeClassifier
mdl = DecisionTreeClassifier(max_leaf_nodes=3, random_state=1)
mdl.fit(X, Y_)
pred = mdl.predict([[9]])
print("Predicted value (DTC): ",pred[0])
print("Accuracy (DTC): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['water resource'], Y_, color='b')
plt.plot(X['water resource'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('water resource')  
plt.ylabel('High/Low') 
plt.show()
print("-------------------------------------")
#LINEAR REGRESSION
mdl = LinearRegression()
mdl.fit(X, Y)
pred = mdl.predict([[7]])
print("Predicted value (LR): ",pred[0])
print("Accuracy (LR): ",mdl.score(X[:200], Y[:200])*100)

plt.scatter(X['water resource'], Y, color='b')
plt.plot(X['water resource'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('water resource')  
plt.ylabel('stable') 
plt.show()

print("-------------------------------------")


#RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
mdl = RandomForestRegressor(n_estimators=100,max_depth=6)
mdl.fit(X, Y)
pred = mdl.predict([[7]])
print("Predicted value (RFR): ",pred[0])
print("Accuracy (RFR): ",mdl.score(X[:200], Y[:200])*100)

plt.scatter(X['water resource'], Y, color='b')
plt.plot(X['water resource'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('water resource')  
plt.ylabel('High/Low') 
plt.show()

print("-------------------------------------")


#DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
mdl =  DecisionTreeRegressor(max_depth=3)
mdl.fit(X, Y)
pred = mdl.predict([[7]])
print("Predicted value (DTR): ",pred[0])
print("Accuracy (DTR): ",mdl.score(X[:300], Y[:300])*100)

plt.scatter(X['water resource'], Y, color='b')
plt.plot(X['water resource'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('water resource')  
plt.ylabel('High/Low')  
plt.show()
print("-------------------------------------")



#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
mdl = LogisticRegression()
mdl.fit(X, Y_)
pred = mdl.predict([[7]])
print("Predicted value (LGR): ",pred[0])
print("Accuracy (LGR): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['water resource'], Y_, color='b')
plt.plot(X['water resource'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('water resource')  
plt.ylabel('stable') 
plt.show()
print("-------------------------------------")


#RANDOM FOREST CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
mdl = RandomForestClassifier(criterion='entropy')
mdl.fit(X, Y_)
pred = mdl.predict([[8]])
print("Predicted value (RFC): ",pred[0])
print("Accuracy (RFC): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['water resource'], Y_, color='b')
plt.plot(X['water resource'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('water resource')  
plt.ylabel('High/Low') 
plt.show()
print("-------------------------------------")


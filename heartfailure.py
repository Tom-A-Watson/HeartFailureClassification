import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
data = pd.read_csv("../../OneDrive/Documents/Year 2/Intelligent Systems 1/heart.csv")

# Checks whether the target label 'HeartDisease' is balanced or unbalanced
print('Amount of patients without a heart disease:', (data['HeartDisease']==0).sum())
print('Amount of patients with a heart disease:', (data['HeartDisease']==1).sum())

# Separates non-numerical and numerical columns of data in the dataset 
# and prints them in lists
non_num_columns = data.select_dtypes(include='object').columns.tolist()
num_columns = data.select_dtypes(exclude='object').columns.tolist()
print('The non-numerical columns are:\n', non_num_columns)
print('The numerical columns are:\n', num_columns)

# Replaces all String data with numerical data and prints it
data['Sex'] = data['Sex'].replace('F',0)# 0 is female
data['Sex'] = data['Sex'].replace('M',1)# 1 is male
data['ChestPainType'] = data['ChestPainType'].replace('ATA',1)
data['ChestPainType'] = data['ChestPainType'].replace('NAP',2)
data['ChestPainType'] = data['ChestPainType'].replace('ASY',3)
data['ChestPainType'] = data['ChestPainType'].replace('TA',4)
data['RestingECG'] = data['RestingECG'].replace('Normal',0)
data['RestingECG'] = data['RestingECG'].replace('ST',1)
data['RestingECG'] = data['RestingECG'].replace('LVH',1)
data['ExerciseAngina'] = data['ExerciseAngina'].replace('N',0)# 0 is No
data['ExerciseAngina'] = data['ExerciseAngina'].replace('Y',1)# 1 is yes
data['ST_Slope'] = data['ST_Slope'].replace('Up',1)
data['ST_Slope'] = data['ST_Slope'].replace('Flat',0)
data['ST_Slope'] = data['ST_Slope'].replace('Down',-1)
print(data)

y = data.iloc[:, -1].values
x = data.iloc[:,0:-1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# dt = DecisionTreeClassifier()
# knn = KNeighborsClassifier()
# svc = svm.SVC()
gbc = GradientBoostingClassifier(n_estimators=200)
rf = RandomForestClassifier(n_estimators=12, criterion='entropy')
mlp = MLPClassifier(hidden_layer_sizes=250, max_iter=500)
eclf1 = VotingClassifier(estimators=[('gbc', gbc), ('rf', rf), ('mlp', mlp)], voting='hard')
eclf2 = VotingClassifier(estimators=[('gbc', gbc), ('rf', rf), ('mlp', mlp)], voting='soft')
eclf3 = StackingClassifier(estimators=[('gbc', gbc), ('rf', rf), ('mlp', mlp)])

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(gbc, x, y, cv=cv)
# print("\nGBC 5-fold cross fold validation accuracy:\n",scores)
# print("\nGBC 5-fold cross fold validation accuracy mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(gbc, x, y, cv=cv, scoring='precision_macro') # For precision result
# print("\nGBC 5-fold cross validation precision:\n",scores)
# print("\nGBC 5-fold cross validation precision mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(rf, x, y, cv=cv)
# print("\nRandom forest 5-fold cross validation accuracy:\n",scores)
# print("\nRandom forest 5-fold cross validation accuracy mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(rf, x, y, cv=cv, scoring='precision_macro') # For precision result
# print("\nRandom forest 5-fold cross validation precision:\n",scores)
# print("\nRandom forest 5-fold cross validation precision mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(mlp, x, y, cv=cv)
# print("\nMLP 5-fold cross validation accuracy:\n",scores)
# print("\nMLP 5-fold cross validation accuracy mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(mlp, x, y, cv=cv, scoring='precision_macro') # For precision result
# print("\nMLP 5-fold cross validation precision:\n",scores)
# print("\nMLP 5-fold cross validation precision mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(eclf1, x, y, cv=cv)
# print("\nEnsemble (hard voting) 5-fold cross validation accuracy:\n",scores)
# print("\nEnsemble (hard voting) 5-fold cross validation accuracy mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(eclf1, x, y, cv=cv, scoring='precision_macro') # For precision result
# print("\nEnsemble (hard voting) 5-fold cross validation precision:\n",scores)
# print("\nEnsemble (hard voting) 5-fold cross validation precision mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(eclf2, x, y, cv=cv)
# print("\nEnsemble (soft voting) 5-fold cross validation accuracy:\n",scores)
# print("\nEnsemble (soft voting) 5-fold cross validation accuracy mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(eclf2, x, y, cv=cv, scoring='precision_macro') # For precision result
# print("\nEnsemble (soft voting) 5-fold cross validation precision:\n",scores)
# print("\nEnsemble (soft voting) 5-fold cross validation precision mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(eclf1, x, y, cv=cv)
# print("\nStackingClassifier 5-fold cross validation accuracy:\n",scores)
# print("\nStackingClassifier 5-fold cross validation accuracy mean:",scores.mean())

# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# scores = cross_val_score(eclf1, x, y, cv=cv, scoring='precision_macro') # For precision result
# print("\nStackingClassifier 5-fold cross validation precision:\n",scores)
# print("\nStackingClassifier 5-fold cross validation precision mean:",scores.mean())

gbc = gbc.fit(x_train,y_train)
y_prediction = gbc.predict(x_test)
print("\nCurrent classifier accuracy and precision:",metrics.accuracy_score(y_test, y_prediction), metrics.precision_score(y_test,y_prediction, average = 'macro'))

# dt = dt.fit(x_train,y_train)
# y_pred = dt.predict(x_test) 
# print("\nDecision tree accuracy and precision:",metrics.accuracy_score(y_test, y_pred), metrics.precision_score(y_test,y_pred, average = 'macro'))

rf = rf.fit(x_train,y_train)
y_pred1 = rf.predict(x_test)
print("\nRandom forest accuracy and precision:",metrics.accuracy_score(y_test, y_pred1), metrics.precision_score(y_test,y_pred1, average = 'macro'))

mlp = mlp.fit(x_train,y_train)
y_pred2 = mlp.predict(x_test)
print("\nMLP accuracy and precision:",metrics.accuracy_score(y_test, y_pred2), metrics.precision_score(y_test,y_pred2, average = 'macro'))

# svc = svc.fit(x_train,y_train)
# y_pred3 = svc.predict(x_test) 
# print("\nSVC accuracy and precision:",metrics.accuracy_score(y_test, y_pred3), metrics.precision_score(y_test,y_pred3, average = 'macro'))

# knn = knn.fit(x_train,y_train)
# y_pred4 = knn.predict(x_test)
# print("\nKNeighborsClassifier tree accuracy and precision:",metrics.accuracy_score(y_test, y_pred4), metrics.precision_score(y_test,y_pred4, average = 'macro'))

eclf1 = eclf1.fit(x_train,y_train)
y_pred5 = eclf1.predict(x_test)
print("\nVotingClassifier (hard voting) accuracy and precision:",metrics.accuracy_score(y_test, y_pred5), metrics.precision_score(y_test,y_pred5, average = 'macro'))

eclf2 = eclf2.fit(x_train,y_train)
y_pred6 = eclf2.predict(x_test)
print("\nVotingClassifier (soft voting) accuracy and precision:",metrics.accuracy_score(y_test, y_pred6), metrics.precision_score(y_test,y_pred6, average = 'macro'))

eclf3 = eclf3.fit(x_train,y_train)
y_pred7 = eclf3.predict(x_test)
print("\nStackingClassifier accuracy and precision:",metrics.accuracy_score(y_test, y_pred7), metrics.precision_score(y_test,y_pred7, average = 'macro'))

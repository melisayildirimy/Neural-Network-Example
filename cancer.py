import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.pipeline import Pipeline
from mlxtend.plotting import plot_learning_curves

#Load the dataset
df = pd.read_csv('breast_data.csv')

#Display descriptive statistics of the dataset
df.describe()

#Extract features and target variable
kriter = df.columns[2:-1]
X = df[kriter]
y = df.diagnosis

#Encode the target variable using LabelEncoder
snf = LabelEncoder()
y = snf.fit_transform(df.diagnosis.values)

#Create a pipeline for preprocessing and classification
arr = Pipeline(steps=[('Veri Önisleme', StandardScaler()), ('Siniflandirma', MLPClassifier())])

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, test_size=0.5)

#Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

#Create an instance of MLPClassifier with specified hyperparameters
mlp = MLPClassifier(max_iter=1000, alpha=0.1, hidden_layer_sizes=(15), activation='logistic', solver='adam', random_state=1, early_stopping=True)

#Fit the MLPClassifier model to the training data
mlp.fit(X_train_scaler, y_train)

#Predict the target variable on the test data
y_pred = mlp.predict(X_test_scaler)
mlp_predictX = mlp.predict_proba(X_test_scaler)[:, 1]

#Print classification report, accuracy, and mean squared error
print(classification_report(y_test, y_pred))
print('MLP Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred) * 100))
print('Eğitim Seti Sonucu: {:.2f}%'.format(mlp.score(X_train_scaler, y_train) * 100))
print('Test Seti Sonucu: {:.2f}%'.format(mlp.score(X_test_scaler, y_test) * 100))
print(mean_squared_error(y_test, y_pred))

#Create a confusion matrix heatmap
etiketX = sorted(df.diagnosis.unique())
plot = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".1f", cmap="cividis", xticklabels=etiketX, yticklabels=etiketX)

#Plot learning curves
plot_learning_curves(X_train, y_train, X_test, y_test, mlp, scoring='accuracy')

#Show the plot
plt.show()
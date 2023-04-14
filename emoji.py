import os
import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from mlxtend.plotting import plot_learning_curves

# Extract data from zip file
with ZipFile('Data.zip', 'r') as f:
    f.extractall()

# Define a function for data preprocessing
def dataProcess(dir_path, etiketX):
    X = []
    etiket = []
    for path in sorted(os.listdir(dir_path)):
        if path != ".DS_Store":
            gorsel = cv.imread(dir_path + path, -1)
            gorsel = cv.resize(gorsel, (100, 100))
            X.append(gorsel)
            etiket.append(etiketX)
    X = np.array(X)
    etiket = np.array(etiket)
    return X, etiket

# Load and preprocess data
angry, ang_lab = dataProcess('Data/angry/', 0)
sad, sad_label = dataProcess('Data/sad/', 1)
happy, hap_label = dataProcess('Data/happy/', 2)
normal, norm_label = dataProcess('Data/normal/', 3)
confused, conf_label = dataProcess('Data/confused/', 4)
surprised, sur_label = dataProcess('Data/surprised/', 5)
X = np.concatenate((angry, sad, happy, normal, confused, surprised))
Y = np.concatenate((ang_lab, sad_label, hap_label, norm_label, conf_label, sur_label))
X = X.reshape(-1, 100 * 100)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0, stratify=Y)

# Initialize and train the MLPClassifier
classF = MLPClassifier(solver='sgd', alpha=1e-5, activation='logistic',
                       learning_rate='invscaling', hidden_layer_sizes=(10), random_state=1, max_iter=500)
classF.fit(X, Y)

# Define a function to calculate accuracy
def accuracy(confusion_matrix):
    diagToplam = confusion_matrix.trace()
    toplamX = confusion_matrix.sum()
    return diagToplam / toplamX

# Make predictions and evaluate model performance
y_pred = classF.predict(X_test)
sınıfAd = ["0", "1", "2", "3", "4", "5"]
np.unique(y_test)
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
print("Accuracy sonucu: ", accuracy(cm))
print(classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=sınıfAd))
print(mean_squared_error(y_test, y_pred))

# Plot confusion matrix and learning curves
temp = sns.heatmap(cm, annot=True, cmap='CMRmap')
temp.set_title('Konfisyon Matrisi')
plot_learning_curves(X_train, y_train, X_test, y_test, classF, scoring='accuracy')
plt.show()

# Classification of Network Connection Security Using ML
This project classifies the network connection between two computers as legit or abnormal, abnormal being an indication of a cyber attack.
# Exploratory Data Analysis
EDA on the dataset was pretty simple as the problem involved binary classification of the network connection. A histogram was used to visualize the number of normal and malicious connections.
# Feature Engineering
The dataset had 44 features in total, one being the target labels (connection security). Initially, a random forest classifier was used to analyze the impact of features on the model performance. Out of 43 input features, only 3 seem to affect the accuracy significantly. The rest of the features were observed to cause overfitting of the model. Hence, such features were dropped from the training procedure.
# Model Selection
2 different model are implemented for classification, random forest and artificial neural network. ANN was implemented Scikit-Learn and Tensorflow, to demonstrate the capabilities of the two python libraries.
# Model Parameters
Model parameters were adjusted based upon initial findings from the earlier random forest classifier and to avoid overfitting, the models were trained up to 98 percent accuracy to keep them robust and flexible towards noise in data. ANN hyperparameters are listed in detail in the notebook.
# Graphs and Plots
Confusion matrices and plots of accuracies and losses are included in the notebook to get an idea of model performance in a quick glance.
# Loading Models
There are 4 models:

random_forest_bin_classifier.pkl -> This model can be loaded using joblib.  
```python
model = joblib.load('random_forest_bin_classifier.pkl')
```

ann_bin_classifier.pkl  
```python
model = joblib.load('ann_bin_classifier.pkl')
```

bin_classifier.h5 -> This model is trained using tensorflow. It can be loaded using keras.  
```python
keras.models.load_models('bin_classifier.h5')
```
bin_classifier.keras 
```python 
keras.models.load_models('bin_classifier.h5')
```
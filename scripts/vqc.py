
# pip install qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms scikit-learn pandas
import time 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler

# Load data
train = pd.read_csv("data/train_preprocessed.csv")
test = pd.read_csv("data/test_preprocessed.csv")

X_train = train.drop(columns=["label"]).values
y_train = train["label"].values
X_test = test.drop(columns=["label"]).values
y_test = test["label"].values

# 4 features 
pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

num_features = X_train.shape[1]

# Feature map + ansatz
feature_map = zz_feature_map(feature_dimension=num_features, reps=2)
ansatz = real_amplitudes(num_qubits=num_features, reps=2)

# Optimizer 
optimizer = COBYLA(maxiter=20)

# Sampler
sampler = Sampler()

# Model
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler
)

# Train
start_train = time.time()
vqc.fit(X_train, y_train)
train_time = time.time() - start_train 

# Test
train_score = vqc.score(X_train, y_train)
test_score = vqc.score(X_test, y_test)

#prediction 
preds = vqc.predict(X_test)

# Extra evaluation metrics
acc = accuracy_score(y_test, preds)  
prec = precision_score(y_test, preds, zero_division=0)  
rec = recall_score(y_test, preds, zero_division=0) 
f1 = f1_score(y_test, preds, zero_division=0) 

# Print results
print("VQC Results")  
print("Train Accuracy:", train_score)  
print("Test Accuracy:", test_score)  
print("Training Time:", round(train_time, 2), "seconds")  

print("Accuracy:", acc)  
print("Precision:", prec)  
print("Recall:", rec)  
print("F1 Score:", f1)  

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
feature_map = zz_feature_map(feature_dimension=num_features, reps=1)
ansatz = real_amplitudes(num_qubits=num_features, reps=1)

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
print("Beginning VQC Training...")
start_train = time.time()
vqc.fit(X_train, y_train)
train_time = time.time() - start_train 
print(f'Training time: {round(train_time)} seconds')

# Test
print("Beginning VQC Scoring...")
start_score = time.time()
train_score = vqc.score(X_train, y_train)
test_score = vqc.score(X_test, y_test)
score_time = time.time() - start_score
print(f'Scoring time: {round(score_time)} seconds')

#prediction 
preds = vqc.predict(X_test)

# Extra evaluation metrics
acc = accuracy_score(y_test, preds)  
prec = precision_score(y_test, preds, zero_division=0)  
rec = recall_score(y_test, preds, zero_division=0) 
f1 = f1_score(y_test, preds, zero_division=0) 

# Print results
print(f'VQC on training dataset (4 features): {train_score:.2f}')
print(f'VQC on test dataset     (4 features): {test_score:.2f}')

print(f"{'Model':<35} | {'Train Score':>10} | {'Test Score':>10}")
print('-' * 62)
print(f"{'VQC, 4 features, ZZFeatureMap':<35} | {train_score:>10.2f} | {test_score:>10.2f}")

print(f"\nAdditional Metrics:")
print(f"Accuracy:  {acc:.2f}")  
print(f"Precision: {prec:.2f}")  
print(f"Recall:    {rec:.2f}")  
print(f"F1 Score:  {f1:.2f}")
import pandas as pd
from sklearn.decomposition import PCA

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
vqc.fit(X_train, y_train)

# Test
score = vqc.score(X_test, y_test)

print("VQC Test Accuracy:", score)
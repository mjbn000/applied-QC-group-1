# -*- coding: utf-8 -*-

# !pip install qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms matplotlib pylatexenc seaborn --quiet

import pandas as pd
import seaborn as sns
import time
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt



algorithm_globals.random_seed = 123

train_df = pd.read_csv("/content/train_preprocessed.csv")
test_df  = pd.read_csv("/content/test_preprocessed.csv")

# Replace "label" with whatever your target column is actually named
train_features = train_df.drop(columns=["label"]).values
train_labels   = train_df["label"].values
test_features  = test_df.drop(columns=["label"]).values
test_labels    = test_df["label"].values

pca = PCA(n_components=4)
train_features = pca.fit_transform(train_features)
test_features  = pca.transform(test_features)

num_features = train_features.shape[1]  # 4

# reps=2 gives a richer kernel than reps=1; expressiveness now comes
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
feature_map.decompose().draw(output='mpl', style='clifford', fold=20)

svc = SVC()
_ = svc.fit(train_features, train_labels)

train_score_c4 = svc.score(train_features, train_labels)
test_score_c4  = svc.score(test_features,  test_labels)
print(f'Classical SVC on the training dataset: {train_score_c4:.2f}')
print(f'Classical SVC on the test dataset:     {test_score_c4:.2f}')

# Build quantum kernel from the feature map (replaces ansatz + optimizer + sampler in VQC)
sampler  = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# Fit QSVC — no iterative optimisation loop, kernel matrix is computed once
qsvc = QSVC(quantum_kernel=quantum_kernel)

start = time.time()
qsvc.fit(train_features, train_labels)
elapsed = time.time() - start
print(f'Training time: {round(elapsed)} seconds')

train_score_qsvc4 = qsvc.score(train_features, train_labels)
test_score_qsvc4  = qsvc.score(test_features,  test_labels)
print(f'QSVC on training dataset (4 features): {train_score_qsvc4:.2f}')
print(f'QSVC on test dataset     (4 features): {test_score_qsvc4:.2f}')

# Re-apply PCA from scratch on the original unscaled data
features_2 = PCA(n_components=2).fit_transform(
    train_df.drop(columns=["label"]).values
)
labels_2 = train_df["label"].values
plt.rcParams['figure.figsize'] = (6, 6)
sns.scatterplot(x=features_2[:, 0], y=features_2[:, 1], hue=train_labels, palette='tab10')

train_f2, test_f2, train_l2, test_l2 = train_test_split(
    features_2, labels_2, train_size=0.8, random_state=algorithm_globals.random_seed
)

svc.fit(train_f2, train_l2)
train_score_c2 = svc.score(train_f2, train_l2)
test_score_c2  = svc.score(test_f2,  test_l2)
print(f'Classical SVC on training dataset (2 features): {train_score_c2:.2f}')
print(f'Classical SVC on test dataset     (2 features): {test_score_c2:.2f}')

num_features_2 = features_2.shape[1]  # 2

# Rebuild feature map for 2 qubits
feature_map_2  = ZZFeatureMap(feature_dimension=num_features_2, reps=2)
fidelity_2     = ComputeUncompute(sampler=Sampler())
quantum_kernel_2 = FidelityQuantumKernel(fidelity=fidelity_2, feature_map=feature_map_2)

qsvc_2 = QSVC(quantum_kernel=quantum_kernel_2)

start = time.time()
qsvc_2.fit(train_f2, train_l2)
elapsed = time.time() - start
print(f'Training time: {round(elapsed)} seconds')

train_score_qsvc2 = qsvc_2.score(train_f2, train_l2)
test_score_qsvc2  = qsvc_2.score(test_f2,  test_l2)
print(f'QSVC on training dataset (2 features): {train_score_qsvc2:.2f}')
print(f'QSVC on test dataset     (2 features): {test_score_qsvc2:.2f}')

print(f"{'Model':<35} | {'Train Score':>10} | {'Test Score':>10}")
print('-' * 62)
print(f"{'Classical SVC, 4 features':<35} | {train_score_c4:>10.2f} | {test_score_c4:>10.2f}")
print(f"{'QSVC, 4 features, ZZFeatureMap':<35} | {train_score_qsvc4:>10.2f} | {test_score_qsvc4:>10.2f}")
print('-' * 62)
print(f"{'Classical SVC, 2 features':<35} | {train_score_c2:>10.2f} | {test_score_c2:>10.2f}")
print(f"{'QSVC, 2 features, ZZFeatureMap':<35} | {train_score_qsvc2:>10.2f} | {test_score_qsvc2:>10.2f}")
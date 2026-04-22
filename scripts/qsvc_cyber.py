# -*- coding: utf-8 -*-

# !pip install qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms scikit-learn pandas bcolors

try:
    import time
    import sys
    import pandas as pd
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.utils import algorithm_globals
    from qiskit.primitives import StatevectorSampler as Sampler
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    from sklearn.decomposition import PCA
    
except ImportError as e:
    print (f"Failed package: {e.name}")
    print ("Make sure you have the required modules installed on your system.")
    print ("Run pip install qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms scikit-learn pandas")
    exit()

'''
This script will compare classical SVC to QSVC analysis of the cybersecurity attack dataset. 




'''
algorithm_globals.random_seed = 123
if len(sys.argv) != 3:
    print("please include the locations of the input csv files.")
    print("ex: qsvc_cyber.py <train> <test>")
    exit()


try:
    train_df = pd.read_csv(sys.argv[1])
except FileNotFoundError:
    print(f"training path: {sys.argv[1]} not found")
    exit()

try:
    test_df  = pd.read_csv(sys.argv[2])
except FileNotFoundError:
    print(f"testing path: {sys.argv[2]} not found")
    exit()

print("Starting script")
print(f"Training Path: {sys.argv[1]}")
print(f"Testing Path: {sys.argv[2]}")

# Replace "label" with whatever the column is actually named -> conveniant because label our actual dataset target lol
train_features = train_df.drop(columns=["label"]).values
train_labels   = train_df["label"].values
test_features  = test_df.drop(columns=["label"]).values
test_labels    = test_df["label"].values

pca = PCA(n_components=4)
train_features = pca.fit_transform(train_features)
test_features  = pca.transform(test_features)

num_features = train_features.shape[1]  # 4

# reps=2 gives a richer kernel than reps=1; 
feature_map = zz_feature_map(feature_dimension=num_features, reps=2)
feature_map.decompose().draw(output='mpl', style='clifford', fold=20)

# kernal creation 
sampler  = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# kernel matrix is computed once
qsvc = QSVC(quantum_kernel=quantum_kernel)

print("Beginning QSVC Training..")
start = time.time()
qsvc.fit(train_features, train_labels)

elapsed = time.time() - start
print(f'Training time: {round(elapsed)} seconds')

print("Beginning QSVC Scoring..")
start = time.time()
train_score_qsvc4 = qsvc.score(train_features, train_labels)
test_score_qsvc4  = qsvc.score(test_features,  test_labels)
elapsed = time.time() - start
print(f'Scoring time: {round(elapsed)} seconds')

print(f'QSVC on training dataset (4 features): {train_score_qsvc4:.2f}')
print(f'QSVC on test dataset     (4 features): {test_score_qsvc4:.2f}')

print(f"{'Model':<35} | {'Train Score':>10} | {'Test Score':>10}")
print('-' * 62)
print(f"{'QSVC, 4 features, ZZFeatureMap':<35} | {train_score_qsvc4:>10.2f} | {test_score_qsvc4:>10.2f}")

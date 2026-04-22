# TO RUN THIS SCRIPT:
# Run the following commands from the main project directory:

# pip install pandas scikit-learn
# python scripts/preprocess.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# NSL-KDD dataset column names as per the standard documentation
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

def preprocess_dataset(input_csv, output_csv, sample_size=100):
    """
    Cleans and prepares NSL-KDD for Quantum Machine Learning.
    Reduces features to 5 key indicators and encodes the label as binary.
    """
    # Load the data (NSL-KDD raw files usually do not have headers)
    df = pd.read_csv(input_csv, names=COLUMNS)
    
    # 1. Feature Reduction: Selecting 5 key numerical features:

    # - duration: length of the connection in seconds
    # - src_bytes: bytes sent from source to destination (volume sent)
    # - dst_bytes: bytes sent from destination to source (volume received)
    # - count: number of connections to the same host in the last 2 seconds (host frequency)
    # - srv_count: number of connections to the same service in the last 2 seconds (service frequency)
    
    selected_features = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
    target = 'label'
    
    df_subset = df[selected_features + [target]].copy()
    
    # 2. Binary Encoding: map 'normal' to 0 and all attack types to 1
    df_subset['label'] = df_subset['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # 3. Scaling: crucial for QML as features are often encoded as rotation 
    # angles
    # Standardizing to zero mean and unit variance prevents feature dominance
    scaler = StandardScaler()
    df_subset[selected_features] = scaler.fit_transform(df_subset[selected_features])
    
    # 4. Downsampling: reducing size for faster execution on Qiskit simulators
    if sample_size < len(df_subset):
        df_subset = df_subset.sample(n=sample_size, random_state=42)
    
    df_subset.to_csv(output_csv, index=False)
    print(f"Saved {len(df_subset)} preprocessed samples to {os.path.basename(output_csv)}")

if __name__ == "__main__":
    # Locates directories relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'nsl-kdd')
    output_dir = os.path.join(script_dir, '..', 'data')

    # Ensure the output 'data' directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # This uses the Train+ and Test+ files from the dataset's directory
    train_input = os.path.join(data_dir, 'KDDTrain+.txt')
    test_input = os.path.join(data_dir, 'KDDTest+.txt')

    train_output = os.path.join(output_dir, 'train_preprocessed.csv')
    test_output = os.path.join(output_dir, 'test_preprocessed.csv')

    # Sample size for the train and test sets can be adjusted
    preprocess_dataset(train_input, train_output, sample_size=1000)
    preprocess_dataset(test_input, test_output, sample_size=200)
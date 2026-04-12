import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Paths from proejct roots
train = pd.read_csv('data/train_preprocessed.csv')
test = pd.read_csv('data/test_preprocessed.csv')

X_train = train.drop('label', axis=1)
y_train = train['label']

X_test = test.drop('label', axis=1)
y_test = test['label']

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, preds)
print("Baseline Accuracy:", acc)


# got baseline accuracy of 0.64
# 0.64 * 25= 16 out of 25 correct
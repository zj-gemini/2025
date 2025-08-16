import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_text


# 1. Create a new sample dataset with three features
# Target depends on 'age' and 'score', 'income' is irrelevant
data = {
    "age": [22, 35, 47, 19, 31, 53, 41, 27, 38, 60, 25, 45, 33, 50, 29],
    "income": [
        40,
        70,
        90,
        25,
        80,
        120,
        95,
        50,
        85,
        130,
        42,
        88,
        76,
        110,
        54,
    ],
    "score": [88, 65, 70, 92, 60, 55, 80, 90, 62, 50, 85, 68, 59, 53, 91],
    # Target: Yes if (age > 40 and score < 75) or (age < 30 and score > 80), else No
    "buys_product": [
        "No",
        "No",
        "Yes",
        "No",
        "No",
        "Yes",
        "No",
        "No",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "No",
        "Yes",
        "Yes",
    ],
}
df = pd.DataFrame(data)

# 2. Prepare the data for training
# We separate the features (X) from the target variable (y).

# 2. Prepare the data for training
# Use all three features
X = df[["age", "income", "score"]]
y = df["buys_product"]

# Convert the categorical target to numerical format (Yes=1, No=0)
# This is a requirement for most machine learning algorithms.
y_encoded = y.apply(lambda x: 1 if x == "Yes" else 0)

# 3. Split the data into training and testing sets
# We use 70% of the data for training the model and 30% for testing its accuracy.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42
)


# 4. Initialize and train a Boosted Decision Tree Classifier (AdaBoost)
# We use AdaBoost with DecisionTreeClassifier as the base estimator.
from sklearn.ensemble import AdaBoostClassifier

base_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf = AdaBoostClassifier(estimator=base_clf, n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# 5. Make predictions and evaluate the model
# The model predicts the outcomes for the test data, and we compare them to the
# actual outcomes to calculate the accuracy.
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# 6. Print the first base decision tree in the AdaBoost ensemble as text in the console
if hasattr(clf, "estimators_") and len(clf.estimators_) > 0:
    tree_text = export_text(clf.estimators_[0], feature_names=list(X.columns))
    print("First Decision Tree in AdaBoost Ensemble:\n")
    print(tree_text)
else:
    print("No base estimators found in AdaBoost ensemble.")

# Model Accuracy: 0.40
# First Decision Tree in AdaBoost Ensemble:

# |--- score <= 67.50
# |   |--- class: 0
# |--- score >  67.50
# |   |--- age <= 44.00
# |   |   |--- age <= 35.00
# |   |   |   |--- class: 0
# |   |   |--- age >  35.00
# |   |   |   |--- class: 0
# |   |--- age >  44.00
# |   |   |--- class: 1

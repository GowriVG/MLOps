from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load Dataset
iris = load_iris()
X = iris.data  # features
y = iris.target  # species labels

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
dump(model, "model.joblib")

print("✅ Model trained and saved successfully as model.joblib")

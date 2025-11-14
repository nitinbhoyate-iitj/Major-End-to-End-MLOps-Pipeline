from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = fetch_olivetti_faces()
X = data.data
y = data.target

# Train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Train Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump((model, X_test, y_test), "savedmodel.pth")

print("Model training complete. savedmodel.pth created.")

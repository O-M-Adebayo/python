# This code uses RandomForestClassifier to make
# predictions on a dataset and also compute 
# the importance of the features on the target variables

# --------------------- BEGINNING OF THE CODE --------------------- #

# Import required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv("dataset.csv")

# Split the data into training and testing sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the performance of the model
confusion_matrix(y_test, predictions)

# Visualize the feature importance
importance = model.feature_importances_

# ------------------------- END OF THE CODE ------------------------- #

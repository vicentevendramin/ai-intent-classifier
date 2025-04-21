import pandas as pd
from src.preprocessing import clean_text, remove_stopwords
from src.features import BoWVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the data
try:
    df = pd.read_csv('data/intencoes.csv', quotechar='"')
except FileNotFoundError:
    print(f"Error: File not found in the 'data' directory. Please ensure the file exists.")
    exit()

# Separate features (messages) and labels (intents)
messages = df['Mensagem do Usuário']
intents = df['Intenção']

# Apply preprocessing to the messages
preprocessed_messages = messages.apply(clean_text)
preprocessed_messages = preprocessed_messages.apply(remove_stopwords)  # Adjust language if nedded
# You can add stemming here if you want:
# from src.preprocessing import apply_stemming
# preprocessed_messages = preprocessed_messages.apply(apply_stemming)

# Initialize the BoWVectorizer
bow_vectorizer = BoWVectorizer()

# Fit the vectorizer to the preprocessed messages and tranform them
features = bow_vectorizer.fit_transform(preprocessed_messages)

# Get the feature names (vocabulary)
feature_names = bow_vectorizer.get_feature_names()
print("Number of features (vocabulary size):", len(feature_names))
print("\nFirst 50 features", feature_names[:50])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, intents, test_size=0.2, random_state=42, stratify=intents)
# If you have few examples in your CSV
# X_train, X_test, y_train, y_test = train_test_split(features, intents, test_size=0.2, random_state=42)

print("\nShape of training features:", X_train.shape)
print("Shape of testing features:", X_test.shape)
print("Shape of training labels:", y_train.shape)
print("Shape of testing labels:", y_test.shape)

# Now you have your data prepared:
# X_train: Numerical features for training
# X_test: Numerical features for testing
# y_train: Corresponding intent labels for training
# y_test: Corresponding intent labels for testing

print("\nData preparation complete. You can now proceed with model training.")

# Initialize the Multinomial Naive Bayes classifier
model = MultinomialNB()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on the test set: {accuracy:.2f}")

print("\nClassification Report on the test set:")
print(classification_report(y_test, y_pred))

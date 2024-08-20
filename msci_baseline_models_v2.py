import os
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Ensure necessary resources are available
nltk.download('stopwords')
nltk.download('wordnet')

# Paths to directories
cleaned_directory = r"C:\Users\joesc\OneDrive - University of Waterloo\MES_Thesis_Data\TSX_ESG_Clean"

# Load all cleaned text files
documents = []
file_paths = []

# I load all cleaned text files from the specified directory
for filename in os.listdir(cleaned_directory):
    if filename.endswith("_clean.txt"):
        cleaned_file_path = os.path.join(cleaned_directory, filename)
        with open(cleaned_file_path, 'r', encoding='utf-8') as file:
            cleaned_text = file.read()
            documents.append(cleaned_text)
            file_paths.append(cleaned_file_path)

# I use TF-IDF to convert the cleaned text into numerical features
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(documents)

# I create fake labels for the purpose of initializing models (since I don't have real labels)
# Here, I will just create some random labels for demonstration purposes
labels = np.random.choice(['SASB', 'GRI', 'Other'], len(documents))
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# I split the data for training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, encoded_labels, test_size=0.2, random_state=42)

# Naive Bayes Model
# I initialize and fit a Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

# Logistic Regression Model
# I initialize and fit a Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# I evaluate the models
print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions, target_names=label_encoder.classes_))

print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions, target_names=label_encoder.classes_))

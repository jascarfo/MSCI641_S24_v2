import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from PyPDF2.errors import PdfReadError

# Ensure necessary resources are available
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize pre-processing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        try:
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
        except PdfReadError as e:
            print(f"Error reading {pdf_path}: {e}")
        return text

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Retain specific special characters and numbers, I tried without this but then the accounting codes were lost
    text = re.sub(r'[^a-zA-Z0-9$%&]', ' ', text)
    # Tokenization
    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Rejoin tokens into a single string
    processed_text = ' '.join(tokens)
    return processed_text

# Paths to directories
pdf_directory = r"C:\Users\joesc\OneDrive - University of Waterloo\MES_Thesis_Data\TSX_ESG"
cleaned_directory = r"C:\Users\joesc\OneDrive - University of Waterloo\MES_Thesis_Data\TSX_ESG_Clean"

# Ensure the cleaned text directory exists
os.makedirs(cleaned_directory, exist_ok=True)

# Load and preprocess all cleaned text files
documents = []
file_paths = []
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        
        # Extract and preprocess text
        text = extract_text_from_pdf(pdf_path)
        if text:  # Only preprocess and save if text extraction was successful
            cleaned_text = preprocess_text(text)
            
            # Save the cleaned text to a .txt file
            cleaned_file_path = os.path.join(
                cleaned_directory,
                filename.replace(".pdf", "_clean.txt")
            )

            with open(cleaned_file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)

            documents.append(cleaned_text)
            file_paths.append(cleaned_file_path)

            print(f"Processed and saved: {cleaned_file_path}")

print("All files processed.")

# Example: Vectorize the text data using both CountVectorizer and TfidfVectorizer for a single file
if documents:
    # Vectorize the text data
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()

    # Example: Converting the single cleaned text to a list as the vectorizers expect a list of texts
    example_document = [documents[0]]

    # Fit and transform the data
    count_vectors = count_vectorizer.fit_transform(example_document)
    tfidf_vectors = tfidf_vectorizer.fit_transform(example_document)

    # Print the shape of the vectors to verify
    print("Count Vectors Shape:", count_vectors.shape)
    print("TF-IDF Vectors Shape:", tfidf_vectors.shape)

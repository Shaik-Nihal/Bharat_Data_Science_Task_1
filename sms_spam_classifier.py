# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download stopwords from nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
data = pd.read_csv("N:\Bharat Intern\spam.csv.csv", encoding='latin-1')

# Clean the dataset by removing unnecessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Preview the dataset
print(data.head())
# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing to the dataset
data['processed_message'] = data['message'].apply(preprocess_text)

# Preview processed data
print(data['processed_message'].head())
# Split the data into features (X) and labels (y)
X = data['processed_message']
y = data['label'].map({'ham': 0, 'spam': 1})  # Map labels to binary

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data, transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Initialize the Naive Bayes classifier
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print the classification report
print(classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam']))


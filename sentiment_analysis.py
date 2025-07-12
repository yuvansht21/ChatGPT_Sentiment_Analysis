# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('punkt')  # Ensure punkt is available


# Step 1: Download NLTK data files (needed only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Load and Preprocess the Data
# Load your dataset (replace 'chatgpt_reviews.csv' with your actual file path)
df = pd.read_csv('chatgpt_reviews.csv')

# Print column names to confirm correct structure
print("Column names:", df.columns)

# Preprocess function: clean text data
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

def preprocess(text):
    if pd.isnull(text):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = tokenizer.tokenize(text)  # Tokenize using RegexpTokenizer
    stop_words = set(stopwords.words('english'))  # Get stopwords
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
    return ' '.join(words)


# Apply preprocessing to all reviews (using 'content' column for reviews)
df['cleaned_review'] = df['content'].apply(preprocess)

# Step 3: Convert Score to Sentiment Labels
# Convert 'score' column (1-5) into sentiment labels: negative, neutral, positive
def map_sentiment(score):
    if score <= 2:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else:
        return 'positive'

df['sentiment'] = df['score'].apply(map_sentiment)

# Step 4: Split Data into Training and Test Sets
X = df['cleaned_review']  # Features (reviews)
y = df['sentiment']  # Target (sentiment labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {len(X_train)} reviews")
print(f"Test data: {len(X_test)} reviews")

# Step 5: Vectorize the Text Data (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to the top 5000 words

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = vectorizer.transform(X_test)

print(f"Shape of training data: {X_train_tfidf.shape}")
print(f"Shape of test data: {X_test_tfidf.shape}")

# Step 6: Train the Sentiment Analysis Model (Naive Bayes)
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

# Step 7: Make Predictions and Evaluate the Model
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'neutral', 'negative'], yticklabels=['positive', 'neutral', 'negative'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Visualize Sentiment Distribution
sentiment_counts = df['sentiment'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Distribution of ChatGPT Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Step 9: Word Cloud Visualization
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['cleaned_review']))

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required resources (only first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Get input from user
text = input("Enter a sentence: ")

# Tokenization
tokens = word_tokenize(text)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

# Display results
print("\nOriginal Text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nStemmed Words:")
print(stemmed_words)

print("\nLemmatized Words:")
print(lemmatized_words)
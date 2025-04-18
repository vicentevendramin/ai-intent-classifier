import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # Or LancasterStemmer, SnowballStemmer

# Ensure you have the stopwords downloaded (run once):
import nltk
nltk.download('stopwords')

def clean_text(text):
    """
    Performs basic text cleaning:
    - Converts to lowercase.
    - Removes pontuaction.
    - Removes numbers.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)     # Remove numbers
    return text


def remove_stopwords_en(text, language='english'):
    """
    Removes stopwords from a text.
    """
    stop_words = set(stopwords.words(language))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


def apply_stemming(text, stemmer_type='porter'):
    """
    Applies stemming (reducing words to their root form) to the words in the text.
    """
    stemmer = None
    if stemmer_type == 'porter':
        stemmer = PorterStemmer()
    # elif stemmer_type == 'lancaster':
    #     stemmer = LancasterStemmer()
    # elif stemmer_type == 'snowball':
    #     stemmer = SnowballStemmer('english') # If your text is in English
    else:
        raise ValueError("Stemmer type not supported.")
    
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)


if __name__ == '__main__':
    example_text = "I want to BUY three apartments quickly, please!"
    cleaned_text = clean_text(example_text)
    print(f"Cleaned text: {cleaned_text}")

    text_without_stopwords = remove_stopwords_en(cleaned_text)
    print(f"Text without stopwords: {text_without_stopwords}")

    stemmed_text = apply_stemming(text_without_stopwords)
    print(f"Text with stemming (Porter): {stemmed_text}")

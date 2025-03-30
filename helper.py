import nltk
from nltk.stem import WordNetLemmatizer
import neattext as nt
nltk.download('wordnet')

def clean_text(text):
    docs = nt.TextFrame(text = text)
    docs.remove_puncts()
    docs.remove_stopwords()
    docs.remove_html_tags()
    docs.remove_special_characters()
    docs.remove_emojis()
    docs.fix_contractions()
    return docs.text

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)
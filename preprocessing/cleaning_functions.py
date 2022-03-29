import nltk
import re

def clean_data(review: str) -> str:
    '''Removes digits and stopwords from a string'''
    stopwords = nltk.corpus.stopwords.words("english")
    review = re.sub(r'\d+', ' ', review)

    review = review.split()
    review = " ".join([word for word in review if not word in stopwords])

    return review

print("hei")
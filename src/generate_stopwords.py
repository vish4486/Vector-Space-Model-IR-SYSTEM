"""
This script downloads the standard English stopword list from NLTK
and saves it to data/stopwords.txt.we must run this once before indexing.
"""

import nltk
from nltk.corpus import stopwords

# to ensure stopwords resource is downloaded
nltk.download("stopwords")

# to save stopwords to file
stopword_list = stopwords.words("english")
with open("data/stopwords.txt", "w") as f:
    for word in stopword_list:
        f.write(word + "\n")

print(f"{len(stopword_list)} stopwords saved to data/stopwords.txt.")

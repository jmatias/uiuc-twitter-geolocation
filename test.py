import data.twitter_user as twuser
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# twitter_users = twuser.load_twitter_users(None, dataset='train')
# twitter_users.sort(key=lambda x: x.username)

# twitter_users2 = twuser.load_twitter_users(None, dataset='train').sort(key=lambda x: x.username)[0:10000]

example_words = ["lol", "@javi_matias", "P_(", "jump", "jumping", "Jump", "loool"]
sentence = "It is important to by very pythonly while you are ||| importing with python..... All pythoners have pythoned poorly at least once."


ps = PorterStemmer()

for w in example_words:
    print(ps.stem(w))

print("*********************")

words = word_tokenize(sentence)
words = [ps.stem(w) for w in words]
sentence = ' '.join(words)

for w in words:
    print(ps.stem(w))


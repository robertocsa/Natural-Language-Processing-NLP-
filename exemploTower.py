# Fonte: https://youtu.be/wjZofJX0v4M?si=a8sQkgJxUgFv7rBh&t=899
# pip install gensim
import gensim.downloader
model=gensim.downloader.load("glove-wiki-gigaword-50")
print(model["tower"])
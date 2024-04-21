from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I am happy to do this")

print(result)
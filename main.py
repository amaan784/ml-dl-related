# load huggin face transofrmers library
from transformers import pipeline 

def sentiment_analysis_1():
  # create a pipleine object and input a task
    classifier = pipeline("sentiment-analysis")

    # apply the classifier with the data we want to test
    result = classifier("I am happy to do this")

    print(result) # will output something like [{'label': 'POSITIVE', 'score': 0.9998675584793091}]
  
  
if __name__ == '__main__':
  sentiment_analysis_1()
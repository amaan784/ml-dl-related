# load huggin face transofrmers library
from transformers import pipeline 

def sentiment_analysis():
    # create a pipleine object and input a task
    # works in 3 steps -
    # 1) pre-pocessing the texts -  applies tokenizer to the model
    # 2) feeds the pre processed text to the model
    # 3) post processing - shows results
    classifier = pipeline("sentiment-analysis")
    

    # apply the classifier with the data we want to test
    result = classifier("I am happy to do this")

    print(result) # will output something like [{'label': 'POSITIVE', 'score': 0.9998675584793091}]

    # trying something negative
    result = classifier("This is really horrible.")
    print(result) # will output something like - [{'label': 'NEGATIVE', 'score': 0.9997618794441223}]
  
if __name__ == '__main__':
  sentiment_analysis()
#   text_generation()
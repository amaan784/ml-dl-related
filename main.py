# load huggin face transofrmers library
from transformers import pipeline 

def sentiment_analysis():
    # create a pipleine object and input the task - sentiment analysis
    # works in 3 steps -
    # 1) pre-pocessing the texts -  applies tokenizer to the model (a default model will be chosen)
    # 2) feeds the pre processed text to the model 
    # 3) post processing - shows results
    classifier = pipeline("sentiment-analysis")

    # apply the classifier with the data we want to test
    sa_1_result = classifier("I am happy to do this")

    print(sa_1_result) # will output something like [{'label': 'POSITIVE', 'score': 0.9998675584793091}]

    # trying something negative
    sa_2_result = classifier("This is really horrible.")
    print(sa_2_result) # will output something like - [{'label': 'NEGATIVE', 'score': 0.9997618794441223}]

def text_generation():
   # create a pipeline and input the task text generation with a specific model to use
   generator = pipeline("text-generation", model="distilgpt2")

   # pass the input data with expectation of the model out to have character limit of 30 and return 2 outputs
   result = generator("This year I really want to win", max_length = 30, num_return_sequences=2)

   print(result)
  
if __name__ == '__main__':
  print("---------Sentiment Analysis-------")
#   sentiment_analysis()
  print("\n---------Text Generation------")
  text_generation()
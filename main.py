# load huggin face transofrmers library
from transformers import pipeline 

def sentiment_analysis():
    # create a pipleine object and input the task - sentiment analysis
    # works in 3 steps -
    # 1) pre-pocessing the texts -  applies tokenizer to the model (a default model will be chosen)
    # 2) feeds the pre processed text to the model 
    # 3) post processing - shows results
    sa_classifier = pipeline("sentiment-analysis")

    # apply the classifier with the data we want to test
    sa_1_result = sa_classifier("I am happy to do this")

    print(sa_1_result) # will output something like [{'label': 'POSITIVE', 'score': 0.9998675584793091}]

    # trying something negative
    sa_2_result = sa_classifier("This is really horrible.")
    print(sa_2_result) # will output something like - [{'label': 'NEGATIVE', 'score': 0.9997618794441223}]

def text_generation():
   # create a pipeline and input the task text generation with a specific model to use
   generator = pipeline("text-generation", model="distilgpt2")

   # pass the input data with expectation of the model output to have character limit of 30 and return 2 outputs
   tg_result = generator("This year I really want to win", max_length = 30, num_return_sequences=2)

   print(tg_result)

def zero_shot_classification():
   # create a pipeline and input the task zero shot classification
   zs_classifier = pipeline("zero-shot-classification")

   zsc_result = zs_classifier("I love watching movies especially bollywood ones",
                              candidate_labels=["cinema", "bussiness", "entertainment"])

   print(zsc_result)
   # will output something like -
   # {'sequence': 'I love watching movies especially bollywood ones', 'labels': ['cinema', 'entertainment', 'bussiness'], 'scores': [0.6243946552276611, 0.3563500940799713, 0.019255323335528374]}
  
if __name__ == '__main__':
  print("---------Sentiment Analysis-------")
  sentiment_analysis()

  print("\n---------Text Generation------")
  text_generation()

  print("\n---------Zero Shot Classification------")
  zero_shot_classification()
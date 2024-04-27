# load huggin face transofrmers library
from transformers import pipeline 

def sentiment_analysis():
    """
        Classifies text as positive or negative
    """
    # create a pipleine object and input the task - sentiment analysis
    # a default pre trained model will be chosen
    sa_classifier = pipeline("sentiment-analysis")

    # apply the classifier with the data we want to test
    sa_1_result = sa_classifier("I am happy to do this")

    print(sa_1_result) # will output something like [{'label': 'POSITIVE', 'score': 0.9998675584793091}]

    # trying something negative
    sa_2_result = sa_classifier("This is really horrible.")
    print(sa_2_result) # will output something like - [{'label': 'NEGATIVE', 'score': 0.9997618794441223}]

def text_generation():
   """
        Completes the input text 
   """
   # create a pipeline and input the task text generation with a specific model to use
   generator = pipeline("text-generation", model="distilgpt2")

   # pass the input data with expectation of the model output to have character limit of 30 and return 2 outputs
   tg_result = generator("This year I really want to win", max_length = 30, num_return_sequences=2)

   print(tg_result)
   # will output something like- 
   # [{'generated_text': 'This year I really want to win a trophy in my race for the WTAI championship, and I love the opportunity I got to face the world'}, 
   # {'generated_text': "This year I really want to win. It should be! That's where I wanted to win the title and win this year.\n\nFor my"}]

def zero_shot_classification():
   """
        Classifies text based on the given labels
   """
   # create a pipeline and input the task zero shot classification
   zs_classifier = pipeline("zero-shot-classification")

   zsc_result = zs_classifier("I love watching movies especially bollywood ones",
                              candidate_labels=["cinema", "bussiness", "entertainment"])

   print(zsc_result)
   # will output something like -
   # {'sequence': 'I love watching movies especially bollywood ones', 'labels': ['cinema', 'entertainment', 'bussiness'], 
   # 'scores': [0.6243946552276611, 0.3563500940799713, 0.019255323335528374]}
  
def fill_mask():
    """
        Will predict missing words in a sentence
    """
    # create a pipleine object and input the task - fill mask
    unmasker = pipeline("fill-mask")

    # will return 2 most likely words because of the top_k parameter
    unmasker_result = unmasker("This course will teach you about <mask> models.", top_k=2)

    print(unmasker_result) 
    # will output something like 
    # [{'score': 0.1989048719406128, 'token': 30412, 'token_str': ' mathematical', 'sequence': 'This course will teach you about mathematical models.'}, 
    # {'score': 0.05367991700768471, 'token': 38163, 'token_str': ' computational', 'sequence': 'This course will teach you about computational models.'}]

def ner():
    """
        Identifies entitites such as persons, organizations or locations in a sentence
    """
    # create a pipleine object and input the task - ner
    ner = pipeline("ner", grouped_entities=True)

    ner_result = ner("My name is Amaan and I work on Earth.")

    print(ner_result) 
    # will output something like 
    # [{'entity_group': 'PER', 'score': 0.99865127, 'word': 'Amaan', 'start': 11, 'end': 16}, 
    # {'entity_group': 'LOC', 'score': 0.99144495, 'word': 'Earth', 'start': 31, 'end': 36}]

def question_answering():
    """
        Extracts answers to a question from a given context
    """
    # create a pipleine object and input the task - question answering
    question_answerer = pipeline("question-answering")

    question_answerer_result = question_answerer(question = "Where do I work?", context = "My name is Amaan and I work on Earth.")

    print(question_answerer_result) 
    # will output something like 
    # {'score': 0.6659677028656006, 'start': 31, 'end': 36, 'answer': 'Earth'}


if __name__ == '__main__':
  print("---------Sentiment Analysis-------")
#   sentiment_analysis()

  print("\n---------Text Generation------")
#   text_generation()

  print("\n---------Zero Shot Classification------")
#   zero_shot_classification()

  print("\n---------Fill Mask------")
#   fill_mask()

  print("\n---------Named Entity Recognition------")
#   ner()

  print("\n---------Question Answering------")
#   question_answering()



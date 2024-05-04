# load hugging face transformers library / api

from transformers import pipeline 

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def sentiment_analysis():
    """
        Classifies text as positive or negative
    """
    # create a pipeline object and input the task - sentiment analysis
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
    # create a pipeline object and input the task - fill mask
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
    # create a pipeline object and input the task - ner
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
    # create a pipeline object and input the task - question answering
    question_answerer = pipeline("question-answering")

    question_answerer_result = question_answerer(question = "Where do I work?", context = "My name is Amaan and I work on Earth.")

    print(question_answerer_result) 
    # will output something like 
    # {'score': 0.6659677028656006, 'start': 31, 'end': 36, 'answer': 'Earth'}

def summarization():
    """
        Creates Summarization of text(s)
    """
    # create a pipeline object and input the task - summarization
    summarizer = pipeline("summarization")

    # not a good input text but can give good summaries
    summarizer_result = summarizer("Databricks is a unified, open analytics platform for building, deploying, sharing, and maintaining enterprise-grade data, analytics, and AI solutions at scale." +
                                   "The Databricks Data Intelligence Platform integrates with cloud storage and security in your cloud account, and manages and deploys cloud infrastructure on your behalf.")

    print(summarizer_result) # not a good input text but can 

def translation():
    """
        Translates text from one language to another
    """
    # create a pipeline object and input the task - translation
    # use model for translating from French to English
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

    translator_result = translator("je suis etudiant")

    print(translator_result) 
    # will output something like 
    # [{'translation_text': "I'm a student."}]

def tokenizer_model_test():
    """
        Tokenizer and Model 

        1) Combining the tokenizer with the sentiment analaysis task in the pipeline

        2) Tokenizing a sample input
    """

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # create a pipeline object and input the task - sentiment analysis 
    # pass the model and tokenizer
    sa_classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # apply the classifier with the data we want to test
    sa_1_result = sa_classifier("I am happy to do this")

    print(sa_1_result) # will output something like [{'label': 'POSITIVE', 'score': 0.9998675584793091}] same as before

    print("-----Tokenizing a sample input------")
    # converts text to a mathametical representation
    # will return a dictionary containing ids (numbers) and attention mask (0 in the attention mask means the attention layer should ignore the token)
    # 101 and 102 in the input id list represent the START and STOP of the input
    sequence = "This is a very helpful framework for development"
    result_tokenizer = tokenizer(sequence) 
    print(result_tokenizer) # will output something like {'input_ids': [101, 2023, 2003, 1037, 2200, 14044, 7705, 2005, 2458, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    # will return the diffrent tokens from the input 
    tokens = tokenizer.tokenize(sequence)
    print(tokens) # will output something like ['this', 'is', 'a', 'very', 'helpful', 'framework', 'for', 'development']

    # gives ids or the mathematical representation to the tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids) # will output something like [2023, 2003, 1037, 2200, 14044, 7705, 2005, 2458]

    # decodes and gives original string back
    decoded_string = tokenizer.decode(ids)
    print(decoded_string) # will output something like  this is a very helpful framework for development


if __name__ == '__main__':
#   print("---------Sentiment Analysis-------")
#   sentiment_analysis()

#   print("\n---------Text Generation------")
#   text_generation()

#   print("\n---------Zero Shot Classification------")
#   zero_shot_classification()

#   print("\n---------Fill Mask------")
#   fill_mask()

#   print("\n---------Named Entity Recognition------")
#   ner()

#   print("\n---------Question Answering------")
#   question_answering()

#   print("\n---------Summarizer------")
#   summarization()

#   print("\n---------Translation------")
#   translation()

    print("\n---------Tokenizer Model------")
    tokenizer_model_test()


import os
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI

# image to text
def image_to_text(image_url):
    """
        Converts an image to text
    """
    # crate a pipeline object and feed the task and model name
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    # get the ouput text
    output_text = image_to_text_model(image_url)[0]['generated_text']

    print(output_text)

    return output_text

# llm 
def generate_story(scenario):
    """
        Use a LLM to generate a story based on a given scenario
    """

    # setting a prompt template 
    prompt_template = """
    You are a story teller.
    You can generate a short story based on a simple narrative.
    The limit for the story is 21 words.

    CONTEXT: {scenario}

    STORY:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["scenario"])

    # making the LLM agent
    story_llm = LLMChain(llm=OpenAI(
        model_name = "gpt-3.5-turbo", temperature=1), 
        prompt=prompt, 
        verbose=True
    )

    # getting the prediction
    story = story_llm.predict(scenario=scenario)

    print(story)

    return story

# text to speech
def text_to_speech():
    pass

if __name__ == '__main__':
    # load env
    load_dotenv(find_dotenv())
    
    HUGGINGFACE_HUB_API_TOKEN = os.getenv('HUGGINGFACE_HUB_API_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    IMAGE_URL = os.getenv('IMAGE_URL')
    
    text_received = image_to_text(IMAGE_URL)

    story = generate_story(text_received)


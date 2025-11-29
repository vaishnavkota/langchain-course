import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import aiplatform
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSequence

load_dotenv()

def main():
    print("Hello from langchain-course!")
    
    info = """Amit Shah is an Indian politician serving as the Union Home Minister of India since May 2019, 
    currently in his second consecutive tenure since June 2024. He is a senior leader of the 
    Bharatiya Janata Party (BJP) and was the party's national president from 2014 to 2020.
    Shah is known for his political acumen and played a crucial role in BJP's significant electoral victories,
    particularly the 2014 and 2019 Lok Sabha elections.
    He led the party's "Mission 300 Par" initiative, surpassing the target of 300 seats in 2019."""

    template = """Given the information {info} ,I want you to create
    1: A short summary about the person in maximum 30 words.
    2: List of 3 major achievements of the person."""

    summary_prompt_template = PromptTemplate(
        input_variables=["info"],
        template=template
    )

    prompt = summary_prompt_template.format(info=info)

    # Initialize the Vertex AI model with a Gemini model
    llm = ChatGoogleGenerativeAI(google_api_key=os.getenv("GEMINI_API_KEY"),temperature=0.1, model="gemini-2.0-flash")
    #llm = ChatOllama(temperature=0.1, model="gemma3:4b")
    #if llm.total_tokens > 1000:  # Example input limit check
    #    print("Prompt too long; truncate it")

    chain= summary_prompt_template | llm
    # Call the model with the formatted prompt
    response = chain.invoke({"info": info})
    
    print("Response:\n",response.content)
    print(type(response))

     # Print token usage metadata if available
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        print("Token usage:")
        print("  input_tokens :", response.usage_metadata.get("input_tokens"))
        print("  output_tokens:", response.usage_metadata.get("output_tokens"))
        print("  total_tokens :", response.usage_metadata.get("total_tokens"))
    else:
        print("No usage metadata returned.")
    
if __name__ == "__main__":
    main()

from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores import Pinecone
import openai
import os
from pydantic import BaseModel
from dotenv import load_dotenv
import pinecone


class Prompt(BaseModel):
    prompt: str


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def configure():
    load_dotenv()


configure()

openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

loader = TextLoader('output.txt')
docs = loader.load()
print(type(docs))
embeddings = OpenAIEmbeddings()

texts = text_splitter.split_documents(docs)

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
)

index_name = 'aidateassistant'

vectordb = Pinecone.from_documents(
    texts,
    embedding=embeddings,
    index_name=index_name,
)


def print_hello():
    print('hello, world')


def get_data(query: str):
    context = vectordb.similarity_search(query, 1)
    system_content = f"You are an assistant that helps people with different places in Almaty. When people asks you to show places, you should show them places which fits the their discription. Here are they places that you chose from: {context}. Your answer should look like this: 'I can recommend the following places: *places, their ratings and descriptions with average checks if available, if no description, just omit it, dont say no description*'."
    user_content = f"Show me places with this description + '{query}'"
    message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        max_tokens=1024,
        temperature=0.7,
        presence_penalty=0,
        frequency_penalty=0.1,
        )

    return response['choices'][0]['message']['content']


@app.post('/generate/')
async def generateAnswer(prompt: Prompt):
    response = get_data(prompt.prompt)
    return response

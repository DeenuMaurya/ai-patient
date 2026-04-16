from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from openai import OpenAI
from prompt import build_prompt

pinecone.init(api_key="YOUR_KEY", environment="us-east-1")

index_name = "ckd-patient"

embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(index_name, embeddings)

client = OpenAI()

def ask_patient(question):
    docs = vectorstore.similarity_search(question, k=3)

    if not docs:
        return "I don't know based on my records."

    context = "\n".join([d.page_content for d in docs])

    prompt = build_prompt(context, question)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
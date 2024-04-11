import pickle as pkl
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import  SeleniumURLLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load the .env file which contains the API keys
load_dotenv()

# Set the API keys as environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']= 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')


# data size 2.06455588
with open('data.pkl', 'rb') as f:
    data = pkl.load(f)

print(len(data))

# Limit the data to 300, because the API key are Expensive!!
data = data[:300]


# Define the llm. Here we just use the ChatOpenAI class from langchain_openai.
llm = ChatOpenAI( temperature=0)

# Typical retrieval prompt.
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    \nQuestion: {input} \nContext: {context} \nAnswer:""")

document_chain = create_stuff_documents_chain(llm, prompt)

result_dict = {}
result_dict['query'] = []
result_dict['answer'] = []
result_dict['sys_ans'] = []

def build_db_from_url(urls):
    """Wrapper function to build the vector db from the urls.

    Args:
        urls (list): List of urls to build the vector db from

    Returns:
        _type_: _description_
    """
    # Load the documents from the urls
    loader = SeleniumURLLoader(urls)
    doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )

    doc_splits = text_splitter.split_documents(doc)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )       
    return vectorstore


print("Starting the loop...")
# Loop through the data and get the answer from the model
# We build the vector db from the urls privuded by each data
# and then use the model to get the answer.
for i, doc_point in enumerate(data):

    # Extract the query, answer and search_results
    query = doc_point['query']
    answer_Y = doc_point['answer']
    docs_url = []
    docs_snippet = []

    # Extract the urls and snippets from the search_results
    search_result_list = doc_point['search_results']
    for item in search_result_list:
        docs_url.append(item['page_url'])
        docs_snippet.append(item['page_snippet'])

    retriever = build_db_from_url(docs_url)
    retriever = retriever.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": query})
    sys_ans = response['answer']
    result_dict['query'].append(query)
    result_dict['answer'].append(answer_Y)
    result_dict['sys_ans'].append(sys_ans)


                                
# Save final_dict as pkl
print("Saving the result_dict...")
with open('result.pkl', 'wb') as f:
    pkl.dump(result_dict, f)
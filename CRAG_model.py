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
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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


# This file implements the Corrective RAG model using the langchain library.
# The paper can be found here: https://arxiv.org/abs/2109.03814
# The original paper has a web search step, but we will use the search results provided in the data.

# Define the llm. Here we just use the ChatOpenAI class from langchain_openai.
llm = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)


# Typical retrieval prompt.
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    Try to answer the question as accurately as possible, that means answering the question in sentence 
    instead single word.
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
    loader = SeleniumURLLoader(
                urls = urls,
                arguments=[
                    "log-level=3",
                ])
   
   
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

# todo: document grader

def hullucination_grader(generation, documents):
    prompt = PromptTemplate(
        template="""You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation} """,
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader.invoke({"documents": documents, "generation": generation})


print("Starting the loop...")
# Loop through the data and get the answer from the model
# We build the vector db from the urls privuded by each data
# and then use the model to get the answer.
for i, doc_point in enumerate(data[:2]):

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

    # Build the vector db from the urls
    retriever = build_db_from_url(docs_url)
    retriever = retriever.as_retriever()

    # Get the relattive document from the retriever.
    doc = retriever.invoke(query)

    # Retrieve the answer from the model.
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get the response from the model
    response = retrieval_chain.invoke({"input": query})

    # Test the hallucination grader
    hullucination_grade = hullucination_grader(documents=doc, generation=answer_Y)

    print(hullucination_grade)
#     sys_ans = response['answer']
#     result_dict['query'].append(query)
#     result_dict['answer'].append(answer_Y)
#     result_dict['sys_ans'].append(sys_ans)


                                
# # Save final_dict as pkl
# print("Saving the result_dict...")
# with open('result.pkl', 'wb') as f:
#     pkl.dump(result_dict, f)
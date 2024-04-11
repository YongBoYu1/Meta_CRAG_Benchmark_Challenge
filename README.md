## The Project
This is a RAG Challenge that posted by Meta. The  RAG QA system takes a question Q as input and outputs an answer A; the answer is generated by LLMs according to information retrieved from external sources, or directly from the knowledge internalized in the model. The answer should provide useful information to answer the question, without adding any hallucination or harmful content such as profanit

**TASK 1: RETRIEVAL SUMMARIZATION**.
In this task, you are provided with up to five web pages for each question. While these web pages are likely, but not guaranteed, to be relevant. The objective of this task is to evaluate the answer generation capabilities of the RAG (Retrieval-Augmented Generation) systems.

To download the data, please see: https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/retrieval-summarization/dataset_files.

To know more about the CRAG challenge, please see: https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024.

# Project Architecture
- **baseline_model.py**: This file contains the base model approach for this challenge. We  loop through the data points in the, build vector db from the urls provided in each data point.

- **dataset_description-v1.md**: The file describe the dataset.

- **.gitignore**: The file ignore the files that only suppose to keep locally. 
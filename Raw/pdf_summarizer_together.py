from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_together import ChatTogether  # NEW: Together.ai model
import textwrap
from tqdm import tqdm  # Import tqdm for progress bar

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")  # Store your Together key in .env

# Set Together API Key
os.environ["TOGETHER_API_KEY"] = api_key

# Load PDF
pdf_path = "examples/33pgs.pdf"  # or your real PDF
loader = PyPDFLoader(pdf_path)
# Load the documents with a progress bar
documents = []
with tqdm(total=1, desc="Loading PDF") as pbar:  # Only one task for loading PDF
    documents = loader.load()
    pbar.update(1)

# Split document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# Display progress for splitting the document
with tqdm(total=len(documents), desc="Splitting document") as pbar:
    docs = text_splitter.split_documents(documents)
    pbar.update(len(docs))

# Load LLM using Together.ai (e.g., Mixtral or LLaMA2)
llm = ChatTogether(
    together_api_key=api_key,
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Or try "meta-llama/Llama-2-70b-chat-hf"
    temperature=0.3
)

# Summarize
chain = load_summarize_chain(llm, chain_type="map_reduce")
# Summarize the document
with tqdm(total=1, desc="Generating summary") as pbar:
    summary = chain.invoke(docs)
    pbar.update(1)

# Debug: Print the summary to check its structure
# print("Summary Output:", summary)

# Ensure summary is a string, usually it's stored under a key like 'output_text' in a dictionary
summary_text = summary.get("output_text", "")  # Use 'output_text' instead of 'text'
if not summary_text:
    print("Summary is empty or not in the expected format.")
else:
    # Define the word limit per line (for example, 150 words per line)
    word_limit = 120

    # Wrap the text to the specified word limit
    wrapped_summary = textwrap.fill(summary_text, width=word_limit)

    # Output
    output_file = "summary.txt"
    with tqdm(total=1, desc="Writing summary to file") as pbar:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(wrapped_summary)
        pbar.update(1)

    print(f"📄 PDF Summary written to {output_file}")

import streamlit as st
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.openrouter import OpenRouter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys from environment
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")

# Setup index and LLM
index = LlamaCloudIndex("frequent-primate-2025-05-07", project_name="Default")

llm = OpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=256,
    context_window=4096,
    model="google/gemma-3-27b-it:free",
)

# Chat interface
st.write("Ask me anything!")

query = st.text_input("Your question:", "")

if st.button("Submit"):
    if query.strip():
        with st.spinner("Generating response..."):
            nodes = index.as_retriever().retrieve(query)
            response = index.as_query_engine(llm=llm).query(query)
        st.success("Response:")
        st.write(response)
    else:
        st.warning("Please enter a question.")

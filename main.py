import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLAMACLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY")

# Setup LLM
llm = OpenRouter(
    api_key=OPENROUTER_API_KEY,
    max_tokens=256,
    context_window=4096,
    model="google/gemma-3-27b-it:free",
)

# Setup LlamaCloud Index
index = LlamaCloudIndex(
    name=os.getenv("LLAMACLOUD_INDEX_NAME"),
    project_name=os.getenv("LLAMACLOUD_PROJECT_NAME"),
    project_id=os.getenv("LLAMACLOUD_PROJECT_ID"),
    organization_id=os.getenv("LLAMACLOUD_ORGANIZATION_ID"),
    api_key=LLAMACLOUD_API_KEY,
)

# Prompt template for QA
qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Below is a question. Respond in the same language as the question.\n"
    "If you do not know the answer based on the provided context, respond with 'Tidak tahu'.\n"
    "Given the context information and not prior knowledge, answer the question: {query_str}\n"
)

# Prompt template for refining answers
refine_prompt_str = (
    "We have the opportunity to refine the original answer (only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better answer the question: {query_str}.\n"
    "Always respond in the same language as the question.\n"
    "If you do not know the answer even with context, respond with 'Tidak tahu'.\n"
    "If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)

# System messages
chat_text_qa_msgs = [
    ("system", "Always answer the question in the same language as the user. If you don't know the answer, say you don't know."),
    ("user", qa_prompt_str),
]

chat_refine_msgs = [
    ("system", "Always refine answers in the same language as the user. If you don't know the answer, say you don't know."),
    ("user", refine_prompt_str),
]

# Create prompt templates
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
refine_template = ChatPromptTemplate.from_messages(chat_refine_msgs)

# Streamlit App
st.title("📚 Chatbot JBCocoa")

on = st.toggle("external")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
if prompt := st.chat_input("Ask me anything!"):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if on:
                # Mode: Use LlamaIndex with context
                query_engine = index.as_query_engine(
                    text_qa_template=text_qa_template,
                    refine_template=refine_template,
                    llm=llm
                )
                response = query_engine.query(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            else:
                # Mode: external
                message = ChatMessage(role="user", content=prompt)
                llm_response = llm.chat([message])
                st.markdown(llm_response)
                st.session_state.messages.append({"role": "assistant", "content": llm_response})

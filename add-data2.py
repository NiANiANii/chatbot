from dotenv import load_dotenv
load_dotenv()
import os

# library
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

# pc = PineconeGRPC(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "document-vectordb"
# pinecone_index = pc.Index(index_name)

vector_store = PineconeVectorStore(index_name=index_name,
                                #    embedding=embed_model,
                                   )

# set up parser
parser = LlamaParse(
    result_type="text",
)

file_extractor = {".pdf": parser}

# Load data from documents
documents = SimpleDirectoryReader(input_dir="documents", file_extractor=file_extractor, recursive=True).load_data()
# index = VectorStoreIndex.from_documents(documents)
from llama_index.core import StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec
os.environ["PINECONE_API_KEY"] = "pcsk_35JF1E_3JUX8AYK2qiHvjoX7n1ehrzUE5uoUzgoSALxjn48hg5mjMpMrnoebVtsMUgPRbs"
api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)
pinecone_index = pc.Index("document-vectordb")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index, sparse_embedding_model=None)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# from llama_index.core import Document
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.extractors import TitleExtractor
# from llama_index.core.ingestion import IngestionPipeline, IngestionCache

# # embeddings
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeEmbeddings

# embeddings = PineconeEmbeddings(model="multilingual-e5-large")

# from llama_index.embeddings.langchain import LangchainEmbedding

# embeddings2 = PineconeEmbeddings(
#     model_name="multilingual-e5-large"
# )
# embed_model = LangchainEmbedding(embeddings2)

# # create the pipeline with transformations
# pipeline = IngestionPipeline(
#     transformations=[
#         SentenceSplitter(chunk_size=25, chunk_overlap=0),
#         TitleExtractor(),
#         # OpenAIEmbedding(),
#     ]
# )

# # run the pipeline
# nodes = pipeline.run(documents=[Document.example()])
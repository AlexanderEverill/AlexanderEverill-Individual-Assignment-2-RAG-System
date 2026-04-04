#Setup
%pip install llama-index
%pip install llama-index-retrievers-bm25

import os

os.environ["OPENAI_API_KEY"] = "sk-proj-..."

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

#Download data
!mkdir -p 'data/paul_graham/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

--2024-07-05 10:10:09--  https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.111.133, 185.199.109.133, 185.199.108.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.111.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 75042 (73K) [text/plain]
Saving to: ‘data/paul_graham/paul_graham_essay.txt’

data/paul_graham/pa 100%[===================>]  73.28K  --.-KB/s    in 0.05s

2024-07-05 10:10:09 (1.36 MB/s) - ‘data/paul_graham/paul_graham_essay.txt’ saved [75042/75042]

#Load Data
from llama_index.core import SimpleDirectoryReader

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

from llama_index.core.node_parser import SentenceSplitter

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)

nodes = splitter.get_nodes_from_documents(documents)

#BM25 retriever + Disk Persistance 
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

# We can pass in the index, docstore, or list of nodes to create the retriever
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=2,
    # Optional: We can pass in the stemmer and set the language for stopwords
    # This is important for removing stopwords and stemming the query + text
    # The default is english for both
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)

BM25S Count Tokens:   0%|          | 0/61 [00:00<?, ?it/s]



BM25S Compute Scores:   0%|          | 0/61 [00:00<?, ?it/s]
                                             
                                             bm25_retriever.persist("./bm25_retriever")

loaded_bm25_retriever = BM25Retriever.from_persist_dir("./bm25_retriever")
Finding newlines for mmindex:   0%|          | 0.00/292k [00:00<?, ?B/s]

#BM25 Retriever + Docstore Persistence
# initialize a docstore to store nodes
# also available are mongodb, redis, postgres, etc for docstores
from llama_index.core.storage.docstore import SimpleDocumentStore

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

# We can pass in the index, docstore, or list of nodes to create the retriever
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=2,
    # Optional: We can pass in the stemmer and set the language for stopwords
    # This is important for removing stopwords and stemming the query + text
    # The default is english for both
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)



BM25S Count Tokens:   0%|          | 0/61 [00:00<?, ?it/s]



BM25S Compute Scores:   0%|          | 0/61 [00:00<?, ?it/s]

from llama_index.core.response.notebook_utils import display_source_node

# will retrieve context from specific companies
retrieved_nodes = bm25_retriever.retrieve(
    "What happened at Viaweb and Interleaf?"
)
for node in retrieved_nodes:
    display_source_node(node, source_length=5000)

retrieved_nodes = bm25_retriever.retrieve("What did the author do after RISD?")
for node in retrieved_nodes:
    display_source_node(node, source_length=5000)

#BM25 Retriever + MetaData Filtering
# Intialize document with some metadata
from llama_index.core import Document

documents = [
    Document(text="Hello, world!", metadata={"key": "1"}),
    Document(text="Hello, world! 2", metadata={"key": "2"}),
    Document(text="Hello, world! 3", metadata={"key": "3"}),
    Document(text="Hello, world! 2.1", metadata={"key": "2"}),
]

# Initialize node parser
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.storage.docstore import SimpleDocumentStore

splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)

# Add nodes to docstore
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

# Define metadata filters
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="key",
            value="2",
            operator=FilterOperator.EQ,
        )
    ],
    condition=FilterCondition.AND,
)

from llama_index.core.response.notebook_utils import display_source_node

from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

retrieved_nodes = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=3,
    filters=filters,  # Add filters here
    stemmer=Stemmer.Stemmer("english"),
    language="english",
).retrieve("Hello, world!")

for node in retrieved_nodes:
    display_source_node(node, source_length=5000)

#Hybrid Retriever with BM25 + Chroma
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("dense_vectors")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(
    docstore=docstore, vector_store=vector_store
)

index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
import nest_asyncio

nest_asyncio.apply()

from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [
        index.as_retriever(similarity_top_k=2),
        BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=2
        ),
    ],
    num_queries=1,
    use_async=True,
)

BM25S Count Tokens:   0%|          | 0/61 [00:00<?, ?it/s]



BM25S Compute Scores:   0%|          | 0/61 [00:00<?, ?it/s]
                                             
nodes = retriever.retrieve("What happened at Viaweb and Interleaf?")
for node in nodes:
    display_source_node(node, source_length=5000)

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine(retriever)

response = query_engine.query("What did the author do after RISD?")
print(response)
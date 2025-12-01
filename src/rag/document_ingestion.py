from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

vectorstore = QdrantVectorstore(location=":memory:")
embedder = ChunkEmbedder(client=OpenAIEmbedder(api_key="YOUR_API_KEY", model_name="text-embedding-3-small"))
vectorstore.create_collection("my_documents",vector_config=[VectorConfig(name="embedding", dimensions=1536)])

pipeline = IngestionPipeline(
    modules=[
        DoclingParser(),
        NodeSplitter(max_char=1024),
        embedder,
    ],
    vector_store=vectorstore,
    collection_name="my_documents"
)

pipeline.run("sample.pdf")

results = vectorstore.search(query_vector = [0.0] * 1536, collection_name="my_documents", k=5)
print(results)
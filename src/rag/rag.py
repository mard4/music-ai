from datapizza.clients.openai import OpenAIClient
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

openai_client = OpenAIClient(
    model="gpt-4o-mini",
    api_key="YOUR_API_KEY"
)

dag_pipeline = DagPipeline()
dag_pipeline.add_module("rewriter", ToolRewriter(client=openai_client, system_prompt="Rewrite user queries to improve retrieval accuracy."))
dag_pipeline.add_module("embedder", OpenAIEmbedder(api_key= "YOUR_API_KEY", model_name="text-embedding-3-small"))
dag_pipeline.add_module("retriever", QdrantVectorstore(host="localhost", port=6333).as_retriever(collection_name="my_documents", k=5))
dag_pipeline.add_module("prompt", ChatPromptTemplate(user_prompt_template="User question: {{user_prompt}}\n:", retrieval_prompt_template="Retrieved content:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}"))
dag_pipeline.add_module("generator", openai_client)

dag_pipeline.connect("rewriter", "embedder", target_key="text")
dag_pipeline.connect("embedder", "retriever", target_key="query_vector")
dag_pipeline.connect("retriever", "prompt", target_key="chunks")
dag_pipeline.connect("prompt", "generator", target_key="memory")

query = "tell me something about this document"
result = dag_pipeline.run({
    "rewriter": {"user_prompt": query},
    "prompt": {"user_prompt": query},
    "retriever": {"collection_name": "my_documents", "k": 3},
    "generator":{"input": query}
})

print(f"Generated response: {result['generator']}")
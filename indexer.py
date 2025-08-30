import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Conexión al servidor remoto
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(anonymized_telemetry=False)
)

# Crear colección
collection = client.get_or_create_collection(name="mini_rag")

# Embeddings con HuggingFace
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Documento de prueba
docs = [
    "La inteligencia artificial es un campo de la informática...",
    "Se aplica en visión por computadora, procesamiento de lenguaje natural...",
    "el autor es ALEX"
]

# Generar embeddings
embeddings = embedder.embed_documents(docs)

# Guardar en Chroma remoto
collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(docs))]
)

print("✅ Documentos guardados en Chroma remoto")

# Probar búsqueda
results = collection.query(
    query_texts=["¿Qué es la inteligencia artificial?"],
    n_results=2
)

print("🔎 Resultados:", results)

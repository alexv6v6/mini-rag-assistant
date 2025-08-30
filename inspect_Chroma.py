import chromadb
from chromadb.config import Settings

# 🔌 Conexión al servidor remoto
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(anonymized_telemetry=False)
)

# 📂 Nombre de la colección que usaste en el indexer
COLLECTION_NAME = "mini_rag"

# Obtener colección
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception as e:
    print(f"❌ No se encontró la colección '{COLLECTION_NAME}'. Error: {e}")
    exit()

print(f"📂 Inspeccionando colección: {COLLECTION_NAME}")

# Contar documentos
count = collection.count()
print(f"📊 Total de documentos: {count}")

if count == 0:
    print("⚠️ La colección está vacía.")
    exit()

# Obtener documentos (paginados si son muchos)
batch_size = 5
for offset in range(0, count, batch_size):
    results = collection.get(
        include=["documents", "metadatas"],
        limit=batch_size,
        offset=offset
    )
    for i, doc in enumerate(results["documents"]):
        doc_id = results["ids"][i]
        meta = results["metadatas"][i]
        preview = doc[:100].replace("\n", " ") + "..." if len(doc) > 100 else doc
        print(f"\n🆔 {doc_id}")
        print(f"📄 Texto: {preview}")
        print(f"🗂️ Metadatos: {meta}")

import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Configuración de la página
st.set_page_config(page_title="Mini RAG Assistant", page_icon="🤖")

st.title("🤖 Mini RAG Assistant")
st.write("Haz preguntas sobre tus documentos indexados en Chroma remoto.")

query = st.text_input("Tu pregunta:")

if query:
    with st.spinner("Procesando..."):

        # Inicializar embeddings con HuggingFace
        st.write("🔹 Inicializando embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.success("✅ Embeddings inicializados")

        # Conectar con Chroma remoto
        st.write("🔹 Conectando a Chroma remoto...")
        client = chromadb.HttpClient(
            host="localhost",  # si corres docker en otra máquina, cambia aquí
            port=8000,
            settings=Settings(anonymized_telemetry=False)
        )

        vectorstore = Chroma(
            client=client,
            collection_name="mini_rag",
            embedding_function=embeddings
        )
        st.success("✅ Conexión con Chroma exitosa")

        # Inicializar retriever
        st.write("🔹 Inicializando retriever...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.success("✅ Retriever listo")

        # Inicializar LLM
        st.write("🔹 Inicializando LLM...")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        st.success("✅ LLM listo")

        # Construir chain RAG
        st.write("🔹 Construyendo chain de RAG...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        st.success("✅ Chain creada")

        # Hacer consulta
        st.write("🔹 Consultando el modelo...")
        result = qa_chain.invoke({"query": query})
        st.success("✅ Consulta completada")

        # Mostrar respuesta
        st.markdown("### Respuesta")
        st.write(result["result"])

        # Mostrar documentos fuente
        st.markdown("### 🔎 Documentos fuente")
        if "source_documents" in result and result["source_documents"]:
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(f"**Documento {i}:**")
                st.write(doc.page_content[:500] + "...")
                st.caption(f"Metadata: {doc.metadata}")
        else:
            st.warning("⚠️ No se encontraron documentos relevantes en Chroma.")

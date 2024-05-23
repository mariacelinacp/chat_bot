# Módulo de carregamento de diretórios da Langchain. Aqui ele carrega os dados da pasta para o Python.
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

# Definição de onde tirar os dados:
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

# Definição de onde carregar os documentos e dividí-los em documentos:
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

# Definição de divisão do texto por número de caracteres (recursive character text splitter):
def split_text(documents: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, #indicação do número de caracteres do fragmento.
        chunk_overlap=100, #indicação do número de caracteres de sobreposição de cada fragmento.
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.") #printou o n° de documentos originais e o n° de fragmentos nos quais ele foi dividido.

    document = chunks[10] # Escolha de um fragmento aleatório como documento.
    print(document.page_content) # Print do conteúdo da página.
    print(document.metadata) # Print dos metadados.

    return chunks

def save_to_chroma(chunks: list):
    # Clear out the database first. # Para limpar todas as versões anteriores do banco de dados antes de executar o script para criar um novo:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents. # Criação de um banco de dados Chroma a partir dos fragmentos. # Necessário conta da OPenAI para usar a função de incorporação).
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH # Criação de um Chroma path e definição disso como um diretório persistente.
    )
    db.persist() # Método persist para forçar a salvar o banco de dados.
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()

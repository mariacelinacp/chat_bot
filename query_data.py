import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

# Modelo para criar um prompt. O {context} são as infos obtidas do banco de dados. A {quesiton} é a própria consulta em si.
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI. #Criação de um analisador de argumentos rápido para poder inserir o texto da consulta na linha de comando.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB. # Para carregar o banco de dados a partir de um caminho (já estabelecido anteriormente).
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB. # Procura o fragmento que melhor corresponda à consulta: passa o texto da consulta como argumento e especifica o n° de resultados que quer recuperar.
    results = db.similarity_search_with_relevance_scores(query_text, k=3) #nesse, caso k=3 são os 3 melhores resultados.
    if len(results) == 0 or results[0][1] < 0.7: # adição de uma verificação: se não houver correspondências ou se a pontuação for menor que 0.7.
        print(f"Unable to find matching results.")
        return

# Busca no banco de dados os trechos relevantes.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) # utilização dos dados obtidos pelo prompt template para formatação do modelo com as chaves obtidas. 
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) 
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
# Após a execução desse trecho do código, será obtida uma única string, que será toda a dica, com todos os fragmentos de informação da consulta feita no início.

# Atribuição da variável "model" ao LLM (objeto complexo) "ChatOpenAI" (criando o prompt) para obter a resposta da pergunta feita.
    model = ChatOpenAI()
    response_text = model.predict(prompt)

# Coletando as referências/fontes do material de origem (utilizando informações dos metadados de cada trecho/fragmento do documento) para responder ao prompt e printando a resposta completa.
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()

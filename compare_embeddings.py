# módulo de carregamento de diretórios de Langchain:
from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation import load_evaluator


def main():
    # Get embedding for a word. # Torna uma palavra em um vetor de incorporação. # Define o o caminho para carregar o banco de dados.
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}") # 7:30 do vídeo, não entendi muito bem, assistir de novo.#################################################################### p/ chamar atenção!
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance") # isso é um avaliador.
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1]) # código para executar uma avaliação.
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()

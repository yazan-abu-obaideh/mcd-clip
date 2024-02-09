import cProfile

from mcd_clip.bike_embedding import clip_embedding_calculator
# noinspection PyUnresolvedReferences
from mcd_clip.embedding.embedding_similarity_optimizer import do_problem

if __name__ == "__main__":
    embedding_calculator = clip_embedding_calculator.ClipEmbeddingCalculatorImpl()
    target_text = "Black bicycle"
    cProfile.run("""do_problem(embedding_calculator.from_text(target_text).reshape((512,)),
                     pop_size=2000,
                     n_generations=20,
                     initialize_from_dataset=True
                     )
    """, filename="run-results/profiling_results.txt")

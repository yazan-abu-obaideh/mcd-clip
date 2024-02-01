import os
import uuid

import pandas as pd

from mcd_clip.bike_embedding import clip_embedding_calculator
from mcd_clip.bike_embedding.embedding_similarity_optimizer import optimize_similarity
from mcd_clip.bike_rendering.parametric_to_image_convertor import ParametricToImageConvertor
from mcd_clip.resource_utils import run_result_path, resource_path

convertor = ParametricToImageConvertor()

# run_uuid = str(uuid.uuid4())
#
# os.mkdir(run_result_path(run_uuid))
#
# embedding_calculator = clip_embedding_calculator.ClipEmbeddingCalculatorImpl()
# target_text = "A huge pitch-black bike without a drop of color"
# cfs = optimize_similarity(embedding_calculator.from_text(target_text).reshape((512,)),
#                           pop_size=200,
#                           n_generations=800,
#                           initialize_from_dataset=False,
#                           sample_from_dataset=False
#                           )
# with open(run_result_path(f"{target_text}_results.csv"), "w") as file:
#     cfs.to_csv(file)

data = pd.read_csv(resource_path('clip_sBIKED_processed.csv'))

n_cfs = len(data)
for i in range(min(6, n_cfs)):
    convertor.to_image(data.iloc[i], path_prefix=f"clips_processed_{i}_")

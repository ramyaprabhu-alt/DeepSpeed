from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig
from deepspeed.model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference
import torch
import deepspeed
config = DeepSpeedInferenceConfig(
                                  hidden_size=5,
                                  intermediate_size = 20,
                                  heads=1,
                                  dtype=torch.float32,
                                  pre_layer_norm = False
                                 )
model = DeepSpeedTransformerInference(config=config)
from numpy import random

x = random.randint(100, size=(1,1,5))
print(x)
model(torch.Tensor(x))
print(deepspeed.__version__)
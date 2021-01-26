from sparktorch import PysparkPipelineWrapper
from pyspark.ml.pipeline import Pipeline, PipelineModel
import torch
import torch.nn as nn

loaded_pipeline = PipelineModel.load('simple_nn')

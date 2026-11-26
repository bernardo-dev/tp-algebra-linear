"""Cenas e construtores de cada tópico da apresentação."""

from src.topics.t01_vectors_matrices_tensors import (
    VectorsMatricesTensorsScene,
    build_vectors_matrices_tensors,
)
from src.topics.t02_linear_transformations import (
    LinearTransformationsScene,
    build_linear_transformations,
)
from src.topics.t03_eigen_symmetric_corners import (
    EigenSymmetricCornersScene,
    build_eigen_symmetric_corners,
)
from src.topics.t04_thresholding import (
    ThresholdingScene,
    build_thresholding,
)

__all__ = [
    "EigenSymmetricCornersScene",
    "LinearTransformationsScene",
    "ThresholdingScene",
    "VectorsMatricesTensorsScene",
    "build_eigen_symmetric_corners",
    "build_linear_transformations",
    "build_thresholding",
    "build_vectors_matrices_tensors",
]

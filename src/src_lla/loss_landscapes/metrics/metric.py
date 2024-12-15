# Original functions by marcellodebernardi as part of loss landscapes library distributed under MIT license

# Incorporated into loss landscape analysis (lla) library without modifications

# This library is distributed under Apache 2.0 license


""" Base classes for model evaluation metrics. """

from abc import ABC, abstractmethod
from src_lla.loss_landscapes.model_interface.model_wrapper import ModelWrapper


class Metric(ABC):
    """ A quantity that can be computed given a model or an agent. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model_wrapper: ModelWrapper):
        pass


class MetricPipeline(Metric):
    """ A sequence of metrics to be computed in order, given a model or an agent. """

    def __init__(self, metrics: list):
        super().__init__()
        self.metrics = metrics

    def __call__(self, model_wrapper: ModelWrapper) -> tuple:
        return tuple([metric(model_wrapper) for metric in self.metrics])

import numpy as np


class LogRatioMetric:
    @staticmethod
    def get_metric(points, target_kde, source_kde):
        return np.log(target_kde.evaluate(points.T) /
                      source_kde.evaluate(points.T))

    @staticmethod
    def get_metric_gradient(points, target_kde, source_kde):
        _, d = points.shape
        return target_kde.gradient(points.T) / np.tile(target_kde.evaluate(points.T).reshape((-1, 1)),
                                                       (1, d)) - source_kde.gradient(points.T) / np.tile(
                source_kde.evaluate(points.T).reshape((-1, 1)), (1, d))

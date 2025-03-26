import csv
from dataclasses import dataclass
import math
from math import log2, sqrt
from pathlib import Path

from gen import Gaussian


@dataclass
class Model:
    components: list[Gaussian]

    def sample(self, datum: float) -> float:
        return sum(dist.sample(datum) for dist in self.components)


@dataclass
class Datum:
    independent: float
    dependent: float


def logstar(n: int) -> float:
    assert n > 0
    norm_const = log2(2.865)
    if n == 1:
        return norm_const
    val = log2(n)
    prev = val
    while val >= 0:
        if (t := log2(prev)) <= 0:
            break
        val += t
        prev = t

    return val + norm_const


def message_length_normal(datum: float, mu: float, sd: float, eps: float) -> float:
    # Pr = eps/[sqrt(2*pi)*sd] exp(-0.5*(x-mu)^2/sd^2)
    return (
        log2(sqrt(2 * math.pi) * sd)
        + ((datum - mu) * (datum - mu)) / (2 * sd * sd)
        - log2(eps)
    )


def message_length_uniform(datum: float, low: float, high: float, eps: float) -> float:
    assert low <= datum <= high
    # Pr = eps / (high - low)
    return log2(high - low) - log2(eps)


def message_length_hypothesis(h: Model) -> float:
    ml = logstar(len(h.components))  # number of components
    # dumb: means are uniformally distributed [0, 100]
    # sd are normally distributed about means with sd of 0.1
    # eps is always 0.005
    # weights are uniformally distributed [0, 1]
    # with some domain knowledge this can be way smarter
    for model in h.components:
        # mean
        ml += message_length_uniform(model.mu, 0, 100, 0.1)
        # sd
        ml += message_length_uniform(model.sigma, 0.0, 5.0, 0.001)
        # weight
        ml += message_length_uniform(model.weight, 0, 1, 0.01)

    return ml


def message_length_data(data: list[Datum], h: Model) -> float:
    ml = logstar(len(data))  # length of the data
    for datum in data:
        mu = h.sample(datum.independent)
        ml += message_length_normal(datum.dependent, mu, 0.05, 0.001)

    return ml

def evaluate_model(data: list[Datum], h: Model) -> float:
    return message_length_hypothesis(h) + message_length_data(data, h)

if __name__ == "__main__":
    with Path("out.csv").open() as data_handle:
        reader = csv.reader(data_handle)
        in_data = [Datum(independent=float(row[0]), dependent=float(row[1])) for row in reader]
    model1 = Model(
        components=[
            Gaussian(weight=0.5, mu=10, sigma=2.5),
            Gaussian(weight=0.5, mu=15, sigma=1),
        ]
    )
    ml_h = message_length_hypothesis(model1)
    ml_d_h = message_length_data(in_data, model1)
    print(f"I(H) = {ml_h} + I(D|H) = {ml_d_h} = {ml_h + ml_d_h}")

    model2 = Model(
        components=[
            Gaussian(weight=0.2, mu=10, sigma=0.1),
            Gaussian(weight=0.8, mu=15, sigma=1.5),
        ]
    )
    ml_h = message_length_hypothesis(model2)
    ml_d_h = message_length_data(in_data, model2)
    print(f"I(H) = {ml_h} + I(D|H) = {ml_d_h} = {ml_h + ml_d_h}")

    model3 = Model(
        components=[
            Gaussian(weight=1.0, mu=15, sigma=2.5),
        ]
    )
    ml_h = message_length_hypothesis(model3)
    ml_d_h = message_length_data(in_data, model3)
    print(f"I(H) = {ml_h} + I(D|H) = {ml_d_h} = {ml_h + ml_d_h}")

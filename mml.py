import csv
from dataclasses import dataclass
import math
from math import log2, sqrt
from pathlib import Path

from gen import Gaussian


@dataclass
class Model:
    components: list[Gaussian]

    def sample(self, datum: float, debug=False) -> float:
        return sum(dist.sample(datum, debug) for dist in self.components)


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
    assert low <= datum <= high, f"{low=} {datum=} {high=}"
    # Pr = eps / (high - low)
    return log2(high - low) - log2(eps)


def message_length_hypothesis(h: Model) -> float:
    ml = logstar(len(h.components))  # number of components
    # dumb: means are uniformally distributed [0, 100]
    # sd are uniformally distributed [0.001, 5.0]
    # with some domain knowledge this can be way smarter
    for model in h.components:
        # mean
        ml += message_length_uniform(model.mu, 0, 100, 0.1)
        # sd
        ml += message_length_uniform(model.sd, 0.001, 50.0, 0.001)

    return ml


def message_length_data(data: list[Datum], h: Model, debug=False) -> float:
    ml = logstar(len(data))  # length of the data
    for datum in data:
        mu = h.sample(datum.independent)
        if debug and abs(datum.independent - 10) < 0.005:
            h.sample(datum.independent, True)
            for c in h.components:
                print(datum.independent, datum.dependent, c.sample(datum.independent))
            print(f"{h=}\ndatum={datum.dependent}, {mu=}, delta={datum.dependent - mu}")
        ml += message_length_normal(datum.dependent, mu, 0.001, 0.001)

    return ml

def evaluate_model(data: list[Datum], h: Model, debug=False) -> float:
    return message_length_hypothesis(h) + message_length_data(data, h, debug)

if __name__ == "__main__":
    with Path("out.csv").open() as data_handle:
        reader = csv.reader(data_handle)
        in_data = [Datum(independent=float(row[0]), dependent=float(row[1])) for row in reader]
    model1 = Model(
        components=[
            Gaussian(mu=10, sd=2.5),
            Gaussian(mu=15, sd=1),
        ]
    )
    ml_h = message_length_hypothesis(model1)
    ml_d_h = message_length_data(in_data, model1)
    print(f"I(H) = {ml_h} + I(D|H) = {ml_d_h} = {ml_h + ml_d_h}")

    model2 = Model(
        components=[
            Gaussian(mu=10, sd=0.1),
            Gaussian(mu=15, sd=1.5),
        ]
    )
    ml_h = message_length_hypothesis(model2)
    ml_d_h = message_length_data(in_data, model2)
    print(f"I(H) = {ml_h} + I(D|H) = {ml_d_h} = {ml_h + ml_d_h}")

    model3 = Model(
        components=[
            Gaussian(mu=15, sd=2.5),
        ]
    )
    ml_h = message_length_hypothesis(model3)
    ml_d_h = message_length_data(in_data, model3)
    print(f"I(H) = {ml_h} + I(D|H) = {ml_d_h} = {ml_h + ml_d_h}")

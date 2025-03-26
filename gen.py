import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random


@dataclass(frozen=True)
class Gaussian:
    weight: float
    mu: float
    sigma: float

    def sample(self, x) -> float:
        a = 1.0 / (self.sigma * math.sqrt(2 * math.pi))
        b = (x - self.mu) ** 2
        c = 2 * self.sigma**2.0
        return self.weight * a * math.exp(-b / c)


@dataclass(frozen=True)
class Data:
    errors: Gaussian
    data: list[Gaussian]


def read_params(raw: str) -> Data | None:
    result = json.loads(raw)
    return Data(
        errors=Gaussian(
            mu=result.get("errors", {}).get("mu", 0.0),
            sigma=result.get("errors", {}).get("sigma", 1.0),
            weight=result.get("errors", {}).get("weight", 1.0),
        ),
        data=[
            Gaussian(
                mu=datum.get("mu", 0.0),
                sigma=datum.get("sigma", 1.0),
                weight=datum.get("weight", 1.0),
            )
            for datum in result.get("data", [])
        ],
    )


def sample(settings: Data, x: float) -> float:
    return sum(dist.sample(x) for dist in settings.data) + random.gauss(
        sigma=settings.errors.sigma
    )


if __name__ == "__main__":
    settings = read_params(Path("data.json").read_text())
    with Path("out.csv").open(mode="w") as out:
        w = csv.writer(out)
        x = 9.0
        while x < 17:
            w.writerow((x, sample(settings, x)))
            x += 0.05

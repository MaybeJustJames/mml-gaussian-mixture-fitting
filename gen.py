import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random


@dataclass(frozen=True, eq=True)
class Gaussian:
    mu: float = 0.0
    sd: float = 1.0

    def sample(self, x, debug=False) -> float:
        a = 1.0 / (self.sd * math.sqrt(2 * math.pi))
        b = (x - self.mu) ** 2
        c = 2 * self.sd**2.0
        return a * math.exp(-b / c)


@dataclass(frozen=True)
class Data:
    errors: Gaussian
    data: list[Gaussian]


@dataclass
class Datum:
    independent: float
    dependent: float


def read_params(raw: str) -> Data | None:
    result = json.loads(raw)
    return Data(
        errors=Gaussian(
            mu=result.get("errors", {}).get("mu", 0.0),
            sd=result.get("errors", {}).get("sd", 1.0),
        ),
        data=[
            Gaussian(
                mu=datum.get("mu", 0.0),
                sd=datum.get("sd", 1.0),
            )
            for datum in result.get("data", [])
        ],
    )


def sample(settings: Data, x: float) -> float:
    return sum(dist.sample(x) for dist in settings.data) + random.gauss(
        sigma=settings.errors.sd
    )

def generate_data(settings) -> list[Datum]:
    data = []
    x = 0.0
    while x <= 100.0:
        data.append(Datum(
            independent=x,
            dependent=sample(settings, x)
        ))
        x += 0.1
    return data


if __name__ == "__main__":
    settings = read_params(Path("data.json").read_text())
    with Path("out.csv").open(mode="w") as out:
        w = csv.writer(out)
        x = 0.0
        while x <= 100:
            w.writerow((x, sample(settings, x)))
            x += 0.1

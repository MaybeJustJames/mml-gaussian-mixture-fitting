import math
from pathlib import Path
import random

from gen import Gaussian, read_params, sample
from mml import Datum, Model, evaluate_model


def _clamp(number: float, low: float, high: float) -> float:
    return min(max(number, low), high)


def perturb_mu(m: Model) -> Model:
    # mu has range [0, 100]
    component_idx = random.choice(range(len(m.components)))
    return Model(
        components=[
            Gaussian(
                mu=component.mu,
                sd=component.sd,
            )
            if idx != component_idx
            else Gaussian(
                mu=_clamp(component.mu + random.choice([0.1, -0.1]), 0.0, 100.0),
                sd=component.sd,
            )
            for idx, component in enumerate(m.components)
        ]
    )


def perturb_all_mus(m: Model) -> Model:
    return Model(
        components=[
            Gaussian(
                mu=_clamp(component.mu + random.choice([0.1, -0.1]), 0.0, 100.0),
                sd=component.sd,
            )
            for component in m.components
        ]
    )


def perturb_sd(m: Model) -> Model:
    # sd must be > 0
    component_idx = random.choice(range(len(m.components)))
    return Model(
        components=[
            Gaussian(
                mu=component.mu,
                sd=component.sd,
            )
            if idx != component_idx
            else Gaussian(
                mu=component.mu,
                sd=_clamp(component.sd + random.choice([0.001, -0.001]), 0.001, 50.0),
            )
            for idx, component in enumerate(m.components)
        ]
    )


def perturb_all_sds(m: Model) -> Model:
    # sd must be > 0
    return Model(
        components=[
            Gaussian(
                mu=component.mu,
                sd=_clamp(component.sd + random.choice([0.001, -0.001]), 0.001, 50.0),
            )
            for component in m.components
        ]
    )

def remove_component(m: Model) -> Model:
    if len(m.components) == 1:
        return m

    remove_idx = random.choice(range(len(m.components)))
    return Model(
        components=[
            component for i, component in enumerate(m.components) if i != remove_idx
        ]
    )


def add_component(m: Model) -> Model:
    addition = Gaussian(
        mu=random.uniform(0.0, 100.0),
        sd=random.uniform(0.001, 50.0),
    )

    return Model(components=[addition, *m.components])


def perturb(m: Model) -> Model:
    perturb_fn = random.choice(
        [
            perturb_mu,
            perturb_all_mus,
            perturb_sd,
            perturb_all_sds,
            remove_component,
            add_component,
        ]
    )
    return perturb_fn(m)

def accept(temperature: float, change: float) -> bool:
    """+ve change in ML is bad, -ve is good."""
    if change < 0:
        return True

    check = random.uniform(0.0, 1.0)
    return check < math.exp(-change / temperature)


def optimize(t_max: float, t_min: float, data: list[Datum], factor: float) -> Model:
    temperature = t_max
    candidate = add_component(Model(components=[]))
    ml = evaluate_model(data, candidate)
    sampled = 1
    while temperature > t_min:
        new_candidate = perturb(candidate)
        sampled += 1
        try:
            new_ml = evaluate_model(data, new_candidate)
        except AssertionError:
            print(candidate, end="\n\n")
            print(new_candidate)
            raise
        change = new_ml - ml
        if accept(temperature, change):
            ml = new_ml
            candidate = new_candidate

        temperature = factor * temperature

    print(f"{sampled=}")
    return candidate

if __name__ == "__main__":
    settings = read_params(Path("data.json").read_text())
    data = []
    x = 0.0
    while x <= 100.0:
        data.append(Datum(
            independent=x,
            dependent=sample(settings, x)
        ))
        x += 0.1

    result = optimize(5000.0, 0.01, data, 0.999)
    print(result, evaluate_model(data, result))

    print("Real data source:")
    print(Model(settings.data), evaluate_model(data, Model(settings.data), debug=True))

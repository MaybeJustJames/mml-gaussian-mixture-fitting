from copy import copy
import itertools
import operator
import random

from gen import Gaussian
from mml import Datum, Model, evaluate_model
from sa import add_component, clamp, remove_component

MUTATION_RATE = 0.1
POP_SIZE = 15


def optimize(iterations: int, data: list[Datum]) -> Model:
    population = initialize(POP_SIZE)
    fitnesses = fitness(population, data)
    for generation in range(iterations):
        elitism = sorted(fitnesses, key=operator.itemgetter(0))[0][1]
        population = sorted(
            [*population, *evolve_population(population, data)],
            key=lambda m: evaluate_model(data, m),
        )[: (POP_SIZE - 1)] + [elitism]
        fitnesses = fitness(population, data)

    return population[0]


def evolve_population(pop: list[Model], data: list[Datum]) -> list[Model]:
    old_pop = pop
    new_pop = []
    while len(old_pop) >= 2:
        parents = roulette_wheel_selection(old_pop, data)
        old_pop = [member for member in old_pop if member not in parents]
        child_1, child_2 = crossover(*parents)
        new_pop.extend((mutate(child_1, MUTATION_RATE), mutate(child_2, MUTATION_RATE)))

    return new_pop


def crossover(p1: Model, p2: Model) -> tuple[Model, Model]:
    (components_c1, components_c2) = _split(
        [
            Gaussian(mu=component.mu, sd=component.sd)
            for component in itertools.chain.from_iterable(
                itertools.zip_longest(p1.components, p2.components)
            )
            if component is not None
        ]
    )

    return (Model(components=components_c1), Model(components=components_c2))


def mutate(m: Model, mutation_rate: float) -> Model:
    result = copy(m)
    if random.uniform(0.0, 1.0) <= mutation_rate:
        result = add_component(result)

    if random.uniform(0.0, 1.0) <= mutation_rate:
        result = remove_component(result)

    for i in range(len(result.components)):
        if random.uniform(0.0, 1.0) <= mutation_rate:
            result.components[i] = Gaussian(
                mu=clamp(result.components[i].mu + random.gauss(), 0.0, 100.0),
                sd=result.components[i].sd,
            )
        if random.uniform(0.0, 1.0) <= mutation_rate:
            result.components[i] = Gaussian(
                mu=result.components[i].mu,
                sd=clamp(result.components[i].sd + random.gauss(), 0.001, 50.0),
            )

    return result


def gen_random_component() -> Gaussian:
    return Gaussian(
        mu=random.uniform(0.0, 100.0),
        sd=random.uniform(0.001, 50.0),
    )


def gen_random_model() -> Model:
    num_components = random.randint(1, 15)
    return Model(components=[gen_random_component() for _ in range(num_components)])


def initialize(popsize: int) -> list[Model]:
    return [gen_random_model() for _ in range(popsize)]


def fitness(pop: list[Model], data: list[Datum]) -> list[tuple[float, Model]]:
    return [(evaluate_model(data, indiv), indiv) for indiv in pop]


def population_fitness(pop: list[Model], data: list[Datum]) -> float:
    fitnesses = fitness(pop, data)
    return min(ml for (ml, _) in fitnesses)


def roulette_wheel_selection(
    pop: list[Model], data: list[Datum]
) -> tuple[Model, Model]:
    fitnesses = sorted(fitness(pop, data), key=operator.itemgetter(0))
    sum_of_fitnesses = sum(ml for ml, _ in fitnesses)

    # sum_weights = sum((max_ml / ml) for ml in mls)
    # print(sum_weights)
    normalized_weights = [ml / sum_of_fitnesses for ml, _ in reversed(fitnesses)]
    # print(f"{normalized_weights=}")
    # print(list(zip(normalized_weights, mls)))
    return tuple(
        random.choices(
            population=[model for _, model in fitnesses],
            weights=normalized_weights,
            k=2,
        )
    )


def _split(xs: list[Gaussian]) -> tuple[list[Gaussian], list[Gaussian]]:
    split_pt = len(xs) // 2
    return (xs[:split_pt], xs[split_pt:])

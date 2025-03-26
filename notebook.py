import marimo

__generated_with = "0.11.28"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import altair as alt

    from gen import generate_data, read_params, sample, Data, Gaussian
    from mml import Datum, Model
    from sa import optimize_with_seed as simulated_annealing
    from ga import optimize as genetic_algorithm
    return (
        Data,
        Datum,
        Gaussian,
        Model,
        Path,
        alt,
        generate_data,
        genetic_algorithm,
        mo,
        pd,
        read_params,
        sample,
        simulated_annealing,
    )


@app.cell
def _(Datum, Model, alt, pd):
    def show(data: list[Datum], sa: Model, ga: Model):
        raw = pd.DataFrame(
            [
                {
                    "x": datum.independent,
                    "y": datum.dependent,
                    "type": "Raw data",
                }
                for datum in data
            ]
        )
        sa_data = pd.DataFrame(
            [
                {
                    "x": datum.independent,
                    "y": sa.sample(datum.independent),
                    "type": "Simulated Annealing",
                }
                for datum in data
            ]
        )
        ga_data = pd.DataFrame(
            [
                {
                    "x": datum.independent,
                    "y": ga.sample(datum.independent),
                    "type": "Genetic Algorithm",
                }
                for datum in data
            ]
        )
        scatter = (
            alt.Chart(raw)
            .mark_circle()
            .encode(
                x=alt.X("x:Q").scale(domain=(0, 100)),
                y=alt.Y("y:Q").scale(domain=(0, 0.5)),
                color=alt.Color("type:N"),
            )
            .properties(width=500, height=500)
        )
        sa_line = (
            alt.Chart(sa_data)
            .mark_line()
            .encode(x=alt.X("x:Q").scale(domain=(0, 100)), y=alt.Y("y:Q").scale(domain=(0, 0.5)), color=alt.Color("type:N"))
        )
        ga_line = (
            alt.Chart(ga_data)
            .mark_line()
            .encode(x=alt.X("x:Q").scale(domain=(0, 100)), y=alt.Y("y:Q").scale(domain=(0, 0.5)), color=alt.Color("type:N"))
        )
        return scatter + sa_line + ga_line
    return (show,)


@app.cell
def _(Data, Gaussian, generate_data):
    #settings = read_params(Path("data.json").read_text())
    settings = Data(errors=Gaussian(sd=0.005), data=[
        Gaussian(mu=55.0, sd=4.9),
        Gaussian(mu=0.0, sd=1.0),
        Gaussian(mu=50.0, sd=1.9),
        Gaussian(mu=77.45, sd=12.7)
    ])
    data = generate_data(settings)
    return data, settings


@app.cell
def _(data, genetic_algorithm):
    seed = genetic_algorithm(100, data)
    seed
    return (seed,)


@app.cell
def _(data, seed, simulated_annealing):
    best = simulated_annealing(100000.0, 0.001, data, 0.999, seed)
    best
    return (best,)


@app.cell
def _(best, data, mo, seed, show):
    chart = mo.ui.altair_chart(show(data, best, seed))
    chart
    return (chart,)


if __name__ == "__main__":
    app.run()

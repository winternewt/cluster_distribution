"""CLI entry point for cluster-distribution."""

import typer

app = typer.Typer(
    name="cluster-distribution",
    help="DBSCAN cluster simulation and Beta-Prime distribution fitting.",
    no_args_is_help=True,
)


@app.command()
def simulate() -> None:
    """Run parallel DBSCAN simulation across an eps sweep."""
    from simulate import main

    main()


@app.command()
def fit() -> None:
    """Fit regular and mixture Beta-Prime models per eps."""
    from beta_mix_vs_regular import main

    main()


@app.command("fit-mixture")
def fit_mixture() -> None:
    """Fit Beta-Prime mixture on merged multi-eps data."""
    from mixure_of_betas import main

    main()


@app.command("fit-merged")
def fit_merged() -> None:
    """Fit eps-dependent Beta-Prime model with regression-linked params."""
    from mixture_of_betas2 import main

    main()


@app.command()
def stats() -> None:
    """Run normality and Poisson goodness-of-fit tests across eps."""
    from stat_tests import main

    main()


@app.command()
def visualize() -> None:
    """Q-Q plots, heatmaps, and regression diagnostics."""
    from visualize import main

    main()


@app.command()
def plot() -> None:
    """Overlay regular vs mixture Beta-Prime PDFs from saved fits."""
    from beta_plot import main

    main()


if __name__ == "__main__":
    app()

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


@app.command()
def demo(
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
    host: str = typer.Option("localhost", "--host", help="Host to bind."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically."),
) -> None:
    """Serve the live CSR demo (webapp/) on a local HTTP server."""
    import os
    import threading
    import webbrowser
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from pathlib import Path
    from functools import partial

    webapp_dir = Path(__file__).parent / "webapp"
    if not webapp_dir.exists():
        typer.echo("webapp/ directory not found.", err=True)
        raise typer.Exit(1)

    Handler = partial(SimpleHTTPRequestHandler, directory=str(webapp_dir))
    server = HTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    typer.echo(f"Serving demo at {url}  (Ctrl-C to stop)")

    if not no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nStopped.")


if __name__ == "__main__":
    app()

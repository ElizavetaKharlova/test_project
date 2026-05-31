"""GraphRec command-line interface."""

from __future__ import annotations

import typer

from graphrec.data.loader import (
    load_movie_features,
    load_movies,
    load_ratings,
    load_user_features,
)
from graphrec.eval.harness import evaluate, recency_ablation
from graphrec.graph.build import build_graph
from graphrec.graph.cache import get_graph
from graphrec.recommend import ppr
from graphrec.recommend.explain import explain as explain_recommendation
from graphrec.recommend.explain import format_explanation

app = typer.Typer(help="Graph-based movie recommender.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Graph-based movie recommender."""


@app.command()
def recommend(
    user: int = typer.Option(..., "--user", "-u", help="MovieLens user id."),
    k: int = typer.Option(10, "--k", "-k", help="Number of recommendations."),
    alpha: float = typer.Option(0.85, help="PageRank damping factor."),
    recency_weight: float = typer.Option(
        0.0, "--recency-weight", help="Tilt toward newer films (0 = off)."
    ),
    explain: bool = typer.Option(
        True, "--explain/--no-explain", help="Show why each item was recommended."
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Force graph reconstruction, ignoring the cache."
    ),
) -> None:
    """Print top-k movie recommendations for a user."""
    movies = load_movies()
    titles = dict(zip(movies["movie_id"], movies["title"], strict=False))

    if recency_weight:
        # A steered graph differs from the cached default, so build it fresh.
        graph = build_graph(
            load_ratings(),
            load_movie_features(),
            load_user_features(),
            recency_weight=recency_weight,
        )
    else:
        graph = get_graph(rebuild=rebuild)
    recs = ppr.recommend(graph, user_id=user, k=k, alpha=alpha, titles=titles)

    typer.echo(f"Top {k} recommendations for user {user}:\n")
    for rank, rec in enumerate(recs, start=1):
        typer.echo(f"{rank:2}. {rec.title}  (score={rec.score:.5f})")
        if explain:
            reason = explain_recommendation(graph, user, rec.movie_id)
            if reason is not None:
                text = format_explanation(reason, titles, rec.title)
                typer.echo(f"      ↳ {text}")


@app.command("eval")
def eval_cmd(
    k: int = typer.Option(10, "--k", help="Cutoff for Recall@K / NDCG@K."),
    test_frac: float = typer.Option(
        0.2, "--test-frac", help="Fraction of the most recent interactions held out."
    ),
    users: int = typer.Option(
        100, "--users", help="Number of sampled eval users (0 = all eligible)."
    ),
    seed: int = typer.Option(42, "--seed", help="Seed for user sampling and SVD."),
    alpha: float = typer.Option(0.85, "--alpha", help="PageRank damping factor."),
) -> None:
    """Compare Graph PPR against baselines on a temporal split."""
    ratings = load_ratings()
    scope = "all eligible" if users == 0 else f"{users} sampled"
    typer.echo(
        f"Evaluating on {scope} users "
        f"(temporal split, test_frac={test_frac}, k={k})...\n"
    )
    table = evaluate(
        ratings,
        movie_features=load_movie_features(),
        user_features=load_user_features(),
        k=k,
        test_frac=test_frac,
        n_users=users,
        seed=seed,
        alpha=alpha,
    )
    typer.echo(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


@app.command("ablate")
def ablate_cmd(
    weights: str = typer.Option(
        "0,0.5,1,2", "--weights", help="Comma-separated recency weights to sweep."
    ),
    k: int = typer.Option(10, "--k", help="Cutoff for Recall@K / NDCG@K."),
    test_frac: float = typer.Option(0.2, "--test-frac", help="Held-out fraction."),
    users: int = typer.Option(100, "--users", help="Number of sampled eval users."),
    seed: int = typer.Option(42, "--seed", help="Seed for user sampling and SVD."),
    alpha: float = typer.Option(0.85, "--alpha", help="PageRank damping factor."),
) -> None:
    """Sweep the recency lever and show its effect on the feature-graph metrics."""
    recency_weights = [float(w) for w in weights.split(",")]
    typer.echo(
        f"Recency ablation on {users} users (sweep={recency_weights}, k={k})...\n"
    )
    table = recency_ablation(
        load_ratings(),
        load_movie_features(),
        load_user_features(),
        recency_weights=recency_weights,
        k=k,
        test_frac=test_frac,
        n_users=users,
        seed=seed,
        alpha=alpha,
    )
    typer.echo(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    app()

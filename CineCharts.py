import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input

from pathlib import Path
import sys

M = 10_000  # smoothing parameter (minimum votes)
STEP = 50_000  # slider step

def prepare_data(base_path: str, m: int = M, step: int = STEP):
    # === Reading files ===
    basics_cache = base_path / "movie.basics.tsv"
    if not basics_cache.is_file():
        movie_basics = (
        pd.read_csv(
            base_path / "title.basics.tsv.gz",
            sep="\t",
            na_values="\\N"
        )
        .query("titleType == 'movie'")
        .drop(columns=["titleType", "endYear"])
        .rename(columns={"startYear": "year"})
    )
        movie_basics.to_csv(base_path / "movie.basics.tsv", sep="\t", index=False)
    else:
        movie_basics = pd.read_csv(base_path / "movie.basics.tsv", sep="\t")

    ratings_cache = base_path / "movie.ratings"
    if not ratings_cache.is_file():
        movie_ratings = (
        pd.read_csv(
            base_path / "title.ratings.tsv.gz",
            sep="\t",
            na_values="\\N"
        )
    .merge(movie_basics[["tconst"]], on="tconst", how="inner")
    )
        movie_ratings.to_csv(base_path / "movie.ratings.tsv", sep="\t", index=False)
    else:
        movie_ratings = pd.read_csv(base_path / "movie.ratings.tsv", sep="\t")


    # === Filtering and derived metrics ===
    # Filter out movies with very few votes (to ensure more reliable ratings)
    movie_ratings = movie_ratings[movie_ratings["numVotes"] >= 1000]

    # IMDb-style weighted rating:
    # WR = (v/(v+m))*R + (m/(v+m))*C
    C = movie_ratings["averageRating"].mean()    # average rating across all movies
    v = movie_ratings["numVotes"].astype(float)  # number of votes per movie
    R = movie_ratings["averageRating"]           # average rating per movie
    movie_ratings["weightedRating"] = (v/(v+m))*R + (m/(v+m))*C


    # Merge ratings with movie basics
    df_all = movie_ratings.merge(movie_basics, on="tconst", how="inner")

    df_runtime = (
    df_all.loc[df_all["runtimeMinutes"].notna(), ["year", "genres", "runtimeMinutes"]]
      .assign(genre=lambda d: d["genres"].str.split(","))
      .explode("genre")
      .drop(columns=["genres"])
      .groupby(["genre", "year"], as_index=False)["runtimeMinutes"].mean()
    )

    # Add direct IMDb link and bubble size for each movie
    df_all["imdb_url"] = "https://www.imdb.com/title/" + df_all["tconst"]
    df_all["bubble_size"] = np.sqrt(df_all["numVotes"])

    # Unique movie genres list
    unique_genres = sorted(df_all["genres"].dropna().str.split(",").explode().unique())

    # Per-genre max (based on the 99th percentile of votes per genre)
    max_votes = genre_quantile_multiple(df_all, unique_genres, step)

    return df_all, df_runtime, unique_genres, max_votes


def genre_quantile_multiple(df: pd.DataFrame, unique_genres: list[str], N: int, q: float =0.99) -> dict[str: int]:
    """
    For each genre from unique_genres, returns the largest number divisible by N
    and less than the quantile value q of the "numVotes" column.
    """
    if not (0 <= q <= 1):
        raise ValueError("q must lie in [0, 1].")
    if N <= 0:
        raise ValueError("N must be positive.")
    
    result: dict[str, int] = {}

    for genre in unique_genres:
        mask = df["genres"].str.contains(rf"\b{genre}\b", na=False)
        if mask.any():
            qval = df.loc[mask, "numVotes"].quantile(q)
            k = (int(qval) // N) * N
            result[genre] = k if k > 0 else 0
        else:
            result[genre] = 0
    
    return result


# === Dash app ===
def create_app(df_all: pd.DataFrame, df_runtime: pd.DataFrame, unique_genres: list[str], max_votes:dict[str, int], m: int = M, step: int = STEP) -> Dash:
    app = Dash(__name__)

    app.layout = html.Div(
        style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto"},
        children=[
            html.H2("CineCharts"),
            dcc.Tabs(id="tabs", value="tab-bubble", children=[
                dcc.Tab(label="Ratings Bubble", value="tab-bubble", children=[
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 2fr", "gap": "16px", "alignItems": "center"},
                        children=[
                            html.Label("Genre:"),
                            dcc.Dropdown(
                                id="genre",
                                options=[{"label": g, "value": g} for g in unique_genres],
                                value=unique_genres[0] if unique_genres else None,
                                clearable=False,
                            ),
                            html.Label("Minimum number of votes:"),
                            dcc.Slider(
                                id="cutoff",
                                min=50_000,
                                max=max(max_votes.values()),
                                step=step,
                                value=100_000,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    dcc.Graph(id="fig", style={"height": "700px"}),
                    html.Div(id="click-info", style={"marginTop": "12px"})
                ]),
                dcc.Tab(label="Runtime by Genre", value="tab-runtime", children=[
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 2fr", "gap": "16px", "alignItems": "start"},
                        children=[
                            html.Label("Genres:"),
                            dcc.Checklist(
                                id="genres-checklist",
                                options=[{"label": g, "value": g} for g in unique_genres],
                                value=[unique_genres[0]] if unique_genres else None,
                                inline=True
                            ),
                        ]
                    ),
                    dcc.Graph(id="duration-fig", style={"height": "700px"}),
                ]),
            ])
        ]
    )

    @app.callback(
        Output("cutoff", "max"),
        Output("cutoff", "marks"),
        Input("genre", "value")
        )
    def update_slider(selected_genre):
        """Adjust slider range and marks to the chosen genre."""
        if not selected_genre:
            return 0, {}
        max_val = max_votes.get(selected_genre, 0)
        marks = {i: str(i) for i in range(0, max_val + 1, step)}
        return max_val, marks

    @app.callback(
        Output("fig", "figure"),
        Input("genre", "value"),
        Input("cutoff", "value"),
    )
    def update_bubble_figure(g, cutoff_value):
        """Update the figure on genre or cutoff change."""
        subset = df_all[(df_all["numVotes"] >= int(cutoff_value)) & (df_all["genres"].str.contains(g, na=False))]
        return make_bubble_figure(df_all, subset, g, int(cutoff_value), m)

    @app.callback(
        Output("click-info", "children"),
        Input("fig", "clickData"),
    )
    def show_click_info(clickData):
        """Show details and a link when a bubble is clicked."""
        if not clickData or "points" not in clickData or not clickData["points"]:
            return ""
        p = clickData["points"][0]
        title = p.get("hovertext", "")
        year = p.get("x", "")
        rating = p.get("y", "")
        num_votes, imdb_url = p.get("customdata", [None, None])
        return html.Div([
            html.B(title),
            html.Span(f" — {year}, rating {rating}, votes {num_votes} — "),
            html.A("Open on IMDb", href=imdb_url, target="_blank", rel="noopener noreferrer")
        ])

    @app.callback(
        Output("duration-fig", "figure"),
        Input("genres-checklist", "value"),
    )
    def update_runtime_figure(selected_genres):
        return make_runtime_figure(df_runtime, selected_genres or [])
    
    return app


def make_bubble_figure(df_all: pd.DataFrame, subset: pd.DataFrame, g: str, cutoff_value: int, m: int) -> go.Figure:
    """Create a bubble chart of movie ratings for a given genre and vote cutoff."""
    subset = subset.copy()
    x_min = (subset["year"].min() // 10) * 10
    x_max = (subset["year"].max() // 10) * 10 + 9

    fig = px.scatter(
        subset,
        x="year",
        y="averageRating",
        size="bubble_size",
        color_discrete_sequence=["royalblue"],
        hover_name="primaryTitle",
        custom_data=["numVotes", "imdb_url"],
        labels={"year": "Release Year", "averageRating": "IMDb Rating"},
        template="plotly_white",
        title=f"Top {len(subset)} Most Popular {g} Films (votes ≥ {cutoff_value})"
    )

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "Year: %{x}<br>" +
                      "Rating: %{y}<br>" +
                      "Votes: %{customdata[0]}<br>" +
                      "<extra></extra>"
    )

    # Per-year average rating for all movies
    avg = df_all.groupby("year")["averageRating"].mean().reset_index()
    avg = avg[(avg["year"] >= x_min) & (avg["year"] <= x_max)]

    fig.add_trace(
        go.Scatter(
            x=avg["year"],
            y=avg["averageRating"],
            mode="lines",
            line=dict(color="darkorange", width=3),
            showlegend=True,
            hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Average: %{y:.2f}<extra></extra>",
            name="Average Rating of all Movies"
        )
    )

    # Per-year IMDb-weighted average for all movies
    imdb_avg = df_all.groupby("year")["weightedRating"].mean().reset_index()
    imdb_avg = imdb_avg[(imdb_avg["year"] >= x_min) & (imdb_avg["year"] <= x_max)]

    fig.add_trace(
        go.Scatter(
            x=imdb_avg["year"],
            y=imdb_avg["weightedRating"],
            mode="lines",
            line=dict(color="forestgreen", width=3),
            showlegend=True,
            hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Average: %{y:.2f}<extra></extra>",
            name=f"IMDb Weighted Rating of all Movies (m = {m})"
        )
    )
    
    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=100, b=75),
    )
    return fig


def make_runtime_figure(df_runtime: pd.DataFrame, selected_genres: list[str]) -> go.Figure:
    """Create a line chart of average movie runtime by genre over the years."""
    if not selected_genres:
        return go.Figure(layout=go.Layout(
            template="plotly_white",
            title="Select at least one genre",
            xaxis=dict(title="Year"),
            yaxis=dict(title="Average runtime (min)")
        ))

    fig = go.Figure(layout=go.Layout(
        template="plotly_white",
        title="Average Movie Runtime by Genre over Years",
        xaxis=dict(title="Year", tickmode="linear", dtick=10),
        yaxis=dict(title="Average runtime (min)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=100, b=75),
    ))

    for g in selected_genres:
        chunk = df_runtime[df_runtime["genre"] == g].sort_values("year")
        if not chunk.empty:
            fig.add_trace(go.Scatter(
                x=chunk["year"],
                y=chunk["runtimeMinutes"],
                mode="lines+markers",
                name=g,
                hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Avg runtime: %{y:.1f} min<extra></extra>"
            ))
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python CineCharts.py <data_path>")
        sys.exit(1) 
    
    if not Path(sys.argv[1]).is_dir():
        print(f"Error: The path '{sys.argv[1]}' is not a valid directory.")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    df_all, df_runtime, unique_genres, max_votes = prepare_data(data_path)
    app = create_app(df_all, df_runtime, unique_genres, max_votes)
    app.run(debug=True)

if __name__ == "__main__":
    main()
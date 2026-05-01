from __future__ import annotations

import argparse
from pathlib import Path

from src.churn_pipeline import clean_data, engineer_features, load_raw_data


def _rate_table(df, column: str):
    return (
        df.groupby(column)["Exited"]
        .agg(churn_rate="mean", customers="size")
        .sort_values("churn_rate", ascending=False)
        .assign(churn_rate=lambda x: (x["churn_rate"] * 100).round(1))
        .reset_index()
    )


def build_dashboard(input_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = engineer_features(clean_data(load_raw_data(input_path)))
    churn_rate = df["Exited"].mean() * 100
    avg_age_churned = df.loc[df["Exited"] == 1, "Age"].mean()
    avg_age_stayed = df.loc[df["Exited"] == 0, "Age"].mean()

    geo_table = _rate_table(df, "Geography").to_html(index=False, classes="data-table")
    gender_table = _rate_table(df, "Gender").to_html(index=False, classes="data-table")
    product_table = _rate_table(df, "NumOfProducts").to_html(index=False, classes="data-table")
    active_table = _rate_table(df, "IsActiveMember").to_html(index=False, classes="data-table")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bank Churn Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2937; background: #f8fafc; }}
    header {{ background: #0f766e; color: white; padding: 28px 36px; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 22px 0; }}
    .metric {{ background: white; border: 1px solid #d9e2ec; border-radius: 6px; padding: 16px; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 20px 0; }}
    button {{ border: 1px solid #0f766e; background: white; color: #0f766e; border-radius: 6px; padding: 9px 12px; cursor: pointer; }}
    button.active {{ background: #0f766e; color: white; }}
    .panel {{ display: none; background: white; border: 1px solid #d9e2ec; border-radius: 6px; padding: 18px; }}
    .panel.active {{ display: block; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #e5e7eb; padding: 10px; text-align: left; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d9e2ec; border-radius: 6px; background: white; }}
  </style>
</head>
<body>
  <header>
    <h1>Bank Customer Churn Dashboard</h1>
    <p>Interactive summary generated from the cleaned churn dataset.</p>
  </header>
  <main>
    <section class="metrics">
      <div class="metric">Customers<strong>{len(df):,}</strong></div>
      <div class="metric">Churn Rate<strong>{churn_rate:.1f}%</strong></div>
      <div class="metric">Avg Age, Churned<strong>{avg_age_churned:.1f}</strong></div>
      <div class="metric">Avg Age, Stayed<strong>{avg_age_stayed:.1f}</strong></div>
    </section>

    <div class="tabs">
      <button class="active" data-panel="geo">Geography</button>
      <button data-panel="gender">Gender</button>
      <button data-panel="products">Products</button>
      <button data-panel="activity">Activity</button>
      <button data-panel="figures">Figures</button>
    </div>

    <section id="geo" class="panel active"><h2>Churn By Geography</h2>{geo_table}</section>
    <section id="gender" class="panel"><h2>Churn By Gender</h2>{gender_table}</section>
    <section id="products" class="panel"><h2>Churn By Number Of Products</h2>{product_table}</section>
    <section id="activity" class="panel"><h2>Churn By Activity Status</h2>{active_table}</section>
    <section id="figures" class="panel">
      <h2>Generated Model Figures</h2>
      <img src="../model_comparison_comprehensive.png" alt="Model comparison">
      <img src="../roc_curves.png" alt="ROC curves">
      <img src="../precision_recall_curves.png" alt="Precision recall curves">
    </section>
  </main>
  <script>
    const buttons = document.querySelectorAll("button[data-panel]");
    const panels = document.querySelectorAll(".panel");
    buttons.forEach((button) => button.addEventListener("click", () => {{
      buttons.forEach((item) => item.classList.remove("active"));
      panels.forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(button.dataset.panel).classList.add("active");
    }}));
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the churn project dashboard.")
    parser.add_argument("--input", default="Churn_Modelling.csv")
    parser.add_argument("--output", default="visualizations/churn_dashboard.html")
    args = parser.parse_args()
    build_dashboard(args.input, args.output)


if __name__ == "__main__":
    main()


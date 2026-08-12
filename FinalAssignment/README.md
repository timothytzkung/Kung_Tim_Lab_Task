# BC Hydro EV Charging & Grid Load Optimization

IAT461 final assignment. Data Scientist: Tim Kung. Client: Phumnawat (Poom) Phosawatmanee.

## Project background

BC Hydro is seeing evening demand spikes in Metro Vancouver as EV owners plug in
after work, straining neighbourhood transformers. Management is weighing three
options:

- **Option A** — a province-wide mandatory peak tariff penalty (4–9 PM) on all
  home power use.
- **Option B** — a capital project to replace/upgrade neighbourhood transformers
  across Metro Vancouver.
- **Option C** — cluster EV drivers into charging-habit archetypes and build
  targeted off-peak incentives.

The assignment evaluates Options A/B/C in EDA, then (assuming Option C is
selected) runs an unsupervised clustering workflow on EV charging session data
to discover behavioural archetypes, which get translated into a rebate/incentive
program for BC Hydro's Demand Side Management team.

Full brief, clarifying Q&A, and the agreed 3-week milestone plan are in
[CONTEXT.md](CONTEXT.md).

## Contents

- **[notebook.ipynb](notebook.ipynb)** / **[notebook.pdf](notebook.pdf)** — the
  main analysis notebook (rendered PDF export alongside the live notebook).
  Runs through:
  - Sessions/consumption/authority-load EDA (time-of-day aggregation, day of
    week & month trends)
  - Policy Options A/B/C evaluation, with a side-by-side summary
  - Feature engineering for clustering (cyclical time encoding,
    energy-to-duration ratio, peak-hour overlap, correlation/redundancy checks)
  - Clustering with K-Means (optimal-k selection, cluster profiles, PCA
    projection) and DBSCAN, plus a K-Means vs. DBSCAN write-up
- **[data/](data)** — input datasets:
  - `sessions.csv` — per-session EV charging records (housing type, charger
    level, start time, duration, energy delivered, peak-overlap ratio, season,
    starting battery state of charge). Primary dataset for the clustering task.
  - `consumption.csv` — BC-wide monthly electricity consumption (MWh).
  - `authority_load.csv` — hourly control-area load data (2025), used for the
    grid load exploration.
- **[reports/week-1-report.pdf](reports/week-1-report.pdf)** — Week 1 milestone
  deliverable (EDA & first clustering pass).
- **[CONTEXT.md](CONTEXT.md)** — client proposal and lab agreement document:
  entity overview, business context, required tasks, stakeholders, clarifying
  Q&A with the client, and the agreed 3-week milestone plan.

## Milestones

Per the lab agreement in CONTEXT.md:

1. **Week 1** — EDA & first clustering pass (see `reports/week-1-report.pdf`)
2. **Week 2** — full draft (refined K-Means vs. DBSCAN comparison, named
   cluster archetypes)
3. **Week 3** — final version (clean notebook, written summary, optional video
   walkthrough)

## Links

Youtube Presentation: [https://youtu.be/5Ai2MdOo2DA](https://youtu.be/5Ai2MdOo2DA)

# Client Proposal: BC Hydro EV Charging & Grid Load Optimization

## 1. Entity Overview

BC Hydro is British Columbia's main electric utility, supplying power to over 5.5 million people. Most of our electricity comes from clean hydro dams, but the fast adoption of Electric Vehicles (EVs) in Metro Vancouver is creating major local grid challenges during evening hours.

## 2. Business Context & Strategic Options

When drivers get home from work and plug in their EVs at the same time, power demand spikes sharply. This surges pressure on neighborhood distribution transformers, risking overheating and outages. Management is considering three ways to handle this:

- **Option A:** Introduce a province-wide mandatory peak tariff penalty on all home power use between 4:00 PM and 9:00 PM.
- **Option B:** Launch a large-scale capital project to replace and upgrade neighborhood transformers across Metro Vancouver.
- **Option C:** Use data science to cluster EV drivers into distinct charging habit groups, allowing us to build targeted off-peak incentives that encourage drivers to shift their charging times.

## 3. Problem Definition

The data scientist assigned to this project needs to analyze Options A, B, and C in EDA, then execute an unsupervised pattern discovery (clustering) workflow on EV charging records.

This is explicitly an unsupervised task. We do not have predefined "good" or "bad" driver labels, nor are we predicting a single target variable. The goal is to discover natural groupings in charging telemetry (start times, energy delivered, charging speed, housing type) so our team can understand how different customer groups use the grid.

## 4. Required Tasks for the Data Scientist

### Phase 1: EDA & Strategy Validation

Analyze `bchydro_ev_charging_sessions.csv` to evaluate Options A, B, and C:

1. Compare load patterns against plug-in times to show why Option A unfairly penalizes low-income non-EV homes and why Option B is too costly.
2. Justify why Option C makes the most sense based on charging habit differences and housing types.

### Phase 2: Data Science Implementation (Unsupervised Clustering)

Assuming management selects Option C:

1. Apply clustering models (e.g., K-Means or DBSCAN) to group sessions into distinct behavioral archetypes (e.g., "Post-Work Heavy Chargers," "Overnight Trickle Chargers").
2. Engineer relevant features (e.g., converting start times to sine/cosine cyclical variables, calculating energy-to-duration ratios, or tracking peak hour overlaps).
3. Evaluate models using metrics like the Silhouette Score to pick the best $k$.
4. Translate cluster statistics into clear business profiles for our Demand Side Management team.

## 5. Stakeholder & Audience

- **Stakeholder:** Director of Grid Modernization & Demand Side Management at BC Hydro.
- **Decision Informed:** Shapes our "EV Smart Charging Incentive Program." Instead of broad rate penalties or expensive upgrades, BC Hydro will use these clusters to offer targeted rebates (e.g., giving "Post-Work Heavy Chargers" a bill credit if they delay charging until 11:00 PM).

---

# Agreement Plan

## IAT461 – Lab Agreement Document

**Project:** BC Hydro EV Charging & Grid Load Optimization
**Data Scientist:** Tim Kung | **Client:** Phumnawat (Poom) Phosawatmanee | **Date:** July 28, 2026

### 1. Purpose

This document records the clarifying questions raised by the data scientist (Tim) about the assigned client (Poom) proposal, the client's answers, the client's stated workload and timeline expectations, and the resulting agreement on scope and milestones for the project.

### 2. Clarification Q&A Log

| Question (Data Scientist → Client) | Answer (Client) |
|---|---|
| The low-income equity argument against Option A – is this derived from data in the dataset, or an assumption? | It is an explanation, not something backed by data in the file. Option A's 4:00–9:00pm penalty applies to all households on the system, so low-income households without an EV would still pay the surcharge despite not driving the behaviour it targets. |
| Should both K-Means and DBSCAN be compared, or should one be selected upfront? | Compare both K-Means and DBSCAN to see which performs better. |
| Should `housing_type` be a clustering input, or used afterward to profile/validate the clusters? | Use `housing_type` directly as a clustering input. No separate post-hoc profiling/validation step is required for it. |
| Beyond Silhouette Score, is there another bar for a "good" clustering result (e.g., cluster sizes, interpretability)? | Yes — also check that cluster sizes are reasonable (no tiny groups) and that the clusters are interpretable/easy to explain in business terms. |

### 3. Workload & Timeline Expectations

- **Expected time commitment:** approximately 20 hours per week.
- **"Done" for the modelling phase:** both K-Means and DBSCAN implemented and compared, with a clear explanation of what each resulting cluster represents.
- **"Done" for the final phase:** a clean, finished notebook plus a short written summary; a 5-minute video walkthrough is a nice-to-have if time allows.
- **Milestone checkpoints:** Week 1 – EDA & first clustering results; Week 2 – full draft; Week 3 – final version.

### 4. Agreed 3-Week Milestone Plan

**Week 1 – EDA & first clustering pass**
Neutral, evidence-based evaluation of Options A/B/C using available fields; the Option A equity point documented explicitly as a stated assumption (not a data-derived finding); feature engineering (cyclical time encoding, energy-to-duration ratio, peak-hour overlap, scaling); `housing_type` included directly as a clustering input; first-pass K-Means and DBSCAN runs with initial Silhouette Score comparison.

**Week 2 – Full draft**
Refined K-Means vs. DBSCAN comparison across Silhouette Score, cluster size balance, and interpretability; named cluster archetypes; full draft notebook and write-up.

**Week 3 – Final version**
Clean, finalized notebook; short written summary translating clusters into incentive recommendations for the Director of Grid Modernization; optional 5-minute video walkthrough.

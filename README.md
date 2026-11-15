# Predictive Maintenance & Digital Twin Dashboard

A machine learning project that predicts how much life an engine has left, detects unusual behavior, and helps plan maintenance.

This project uses the NASA C-MAPSS turbofan engine dataset to train a model and shows the results in a simple dashboard.

## What this project does

- Predicts **Remaining Useful Life (RUL)** — how many more cycles an engine can run
- Detects **unusual engine behavior** using anomaly detection
- Shows the **health of all machines** in one dashboard
- Displays **sensor trends, alerts, and maintenance suggestions**

## How it works

```text
C-MAPSS Engine Data
        │
        ▼
   Train ML Model
        │
        ▼
  Simulator creates
   live engine data
        │
        ▼
   Scoring Engine
   predicts RUL &
   checks health
        │
        ▼
      Dashboard

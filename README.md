# Predicting Vulnerability to Poverty in Colombia using ELCA

## Overview
This project develops a fully reproducible pipeline to predict transitions into monetary poverty using the ELCA dataset.

## Research Question
How well can different predictive models anticipate transitions into monetary poverty among Colombian households?

## Data
Data is obtained from DataHub Uniandes (Dataverse) using API-based automated downloads.

## Reproducibility
The project is fully reproducible:
- Raw data is programmatically downloaded
- Data processing is scripted
- Models are fully replicable

## Project Structure
- `src/`: scripts organized by pipeline stage
- `data/`: raw, interim, processed
- `docs/`: documentation and decisions
- `outputs/`: tables, figures, models
- `paper/`: manuscript

## Pipeline
1. Download data
2. Build datasets per round
3. Clean and harmonize
4. Construct outcome and features
5. Train models
6. Evaluate performance

## Status
Project initialized with data download pipeline and reproducibility structure.
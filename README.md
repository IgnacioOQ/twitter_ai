# Twitter AI Dataset Analysis

This repository contains data analysis pipelines, notebooks, and AI agent definitions for analyzing a Twitter dataset related to artificial intelligence.

## Overview

The primary focus of this project is processing, cleaning, and analyzing Twitter data using Jupyter Notebooks. The repository also includes various Markdown-based definitions and guidelines for AI agents that interact with or process this data.

### Important Notes
- **Google Colab & Google Drive Integration**: The whole project runs and stores its database on Google Colab and Google Drive. To run the notebooks locally, you will need to map your Google Drive to your local machine (e.g., using Google Drive for Desktop) so the paths correctly resolve to the shared data. Alternatively, you can upload the notebooks to Google Colab and execute them there.

## Directory Structure

- `notebooks/`: Jupyter Notebooks for data mining, data processing, topic modeling, and network generation.
- `agents/`: AI agent definitions, protocols, and logs (e.g., MC Agent, MCP Agent).
- `docs/`: Markdown conventions, overarching explanations, and architectural plans.

## Notebooks Workflow
The data pipeline is divided into five sequential stages, each housed in its own subdirectory within `notebooks/`:

1. **`01_Ingestion/`**: Data mining from the Twitter API and access to shared folders.
2. **`02_Processing/`**: Converting raw API data into usable dictionaries, structural sanity checks, network generation, author corpus creation, and text cleaning.
3. **`03_Analysis_and_Modeling/`**: Adding sentiment analysis, extracting examples, LDA topic modeling, and embedding mappings.
4. **`04_Network_Analysis/`**: Pure network graph analysis.
5. **`05_Experiments/`**: Isolated testing (e.g., TP-Bigrams).

## Usage
Since the core logic is in standard Jupyter `.ipynb` format, you can open them with VS Code, JupyterLab, or locally connect them to Google Colab to leverage cloud compute. Ensure that the required python libraries (pandas, scikit-learn, networkx, etc.) are installed if running locally.

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
1. `0.` Data mining and access to shared folders.
2. `1.` Converting raw API data into usable dictionaries.
3. `2.` Sanity checks, network generation, and author corpus creation.
4. `3.` Adding sentiment and topic modeling (LDA) to tweets, grouping by author.
5. `4.` Network analysis.
6. `5.` Embedding mappings.

## Usage
Since the core logic is in standard Jupyter `.ipynb` format, you can open them with VS Code, JupyterLab, or locally connect them to Google Colab to leverage cloud compute. Ensure that the required python libraries (pandas, scikit-learn, networkx, etc.) are installed if running locally.

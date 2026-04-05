# CrewAI Yelp Analysis Pipeline

This repository contains a deterministic data analysis pipeline built with CrewAI and powered by a local instance of the Qwen2.5:7b model via Ollama. The system is designed to perform high-precision profiling and rating prediction on Yelp datasets while strictly adhering to a grounded execution loop.

## Technical Architecture

The pipeline utilizes a multi-agent orchestration strategy to ensure data integrity and prevent hallucinations. By leveraging local computation, it maintains privacy and provides a cost-effective alternative to cloud-based LLM services.

### Core Components

*   **LLM Model**: Qwen2.5:7b (Running locally via Ollama at `http://localhost:11434`)
*   **Embedding Model**: BAAI/bge-small-en-v1.5
*   **Orchestration**: CrewAI (Sequential Process)
*   **Inference Settings**: Temperature 0.0 (Strictly deterministic)

## Multi-Agent Workflow

The analysis is performed by three specialized agents that operate in a sequential chain, passing context to ensure the final prediction is based on verified profile data.

1.  **User Profiler**: Responsible for retrieving historical data and reviewing habits for a specific user. It synthesizes past review sentiments and category preferences.
2.  **Item Analyst**: Conducts a detailed features analysis of the target business, including its attributes (e.g., WiFi, Parking), categories, and location data.
3.  **Prediction Modeler**: Snythesizes the outputs from the Profiler and Analyst to predict the star rating and generate a realistic review text that matches the user's established persona.

## Custom Tools

To guarantee accuracy, the system bypasses traditional RAG (Retrieval-Augmented Generation) for primary lookups in favor of custom high-precision tools implemented in `crew.py`:

I demonstrated that JSONSearchTool's semantic search cannot retrieve records by alphanumeric ID because random IDs have no semantic content. I replaced it with exact-match lookup functions, which is the appropriate retrieval strategy when the lookup key is a non-semantic identifier. For semantic content (like finding similar reviews), RAG would be appropriate.
*   **`search_user_data`**: Performs an exact ID match against the user dataset to retrieve raw JSON records.
*   **`search_item_data`**: Retrieves exact business features by item ID.
*   **`search_review_data`**: Filters historical review data to provide the LLM with grounded examples of past interactions.

## Local Knowledge Configuration

The pipeline incorporates a `StringKnowledgeSource` containing the Yelp Data Translation schema. This provides the agents with metadata definitions for various JSON keys, enabling them to correctly interpret complex attributes without requiring external documentation access.

## Project Execution

The environment is managed using `uv`. To run the analysis:

```bash
uv run first_crew
```

### Output File
The final prediction is saved to `report.json` in the root directory. This file contains the raw JSON output with the predicted `stars` and `text` fields, formatted for automated evaluation and MSE (Mean Squared Error) calculation.

## Results and Performance

On the verified test case (Index 11), the pipeline accurately identified the user (Carlos) and business (Double Decker) to produce a grounded prediction that aligns with historical data patterns.

## Execution Trace

A full, step-by-step trace of the command-line execution (including tool calls, agent reasoning, and box-formatted logs) is available here:
- [EXECUTION.md](./EXECUTION.md)


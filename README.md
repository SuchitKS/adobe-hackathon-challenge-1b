# Persona-Driven Document Intelligence System

[cite_start]This repository contains a CPU-only, Dockerized Python solution for extracting and ranking document sections from PDFs based on a user persona and a specific task[cite: 1, 3].

## Features

-   [cite_start]Extracts and ranks relevant sections from multiple PDF documents.
-   [cite_start]Uses a defined **persona** and **"job-to-be-done"** to tailor content relevance.
-   [cite_start]Ranks information using **TF-IDF** and **Cosine Similarity** for contextual accuracy.
-   [cite_start]Identifies sections using regex-based header detection.
-   [cite_start]Offline and lightweight, with no external API calls for processing[cite: 1, 2].
-   [cite_start]Generates a structured JSON output with top sections and refined subsections.

---

## 📁 Folder Structure

.
├── main.py          # Main script for processing, ranking, and output generation 

├── Dockerfile       # Offline, CPU-only Docker environment 

├── requirements.txt   # Python dependencies 

├── input/           # Place your challenge1b_input.json here 

├── output/          # Outputs challenge1b_output.json here 

└── Collection 1/    # Example PDF collection folder 


└── PDFs/



## Approach

-   **Persona Profiling**: Creates a detailed keyword profile from the user's role and task to guide the analysis.
-   **Text & Section Extraction**: Uses `PyPDF2` to extract raw text and regular expressions to identify and segment content into logical sections based on common heading patterns.
-   **TF-IDF Vectorization**: Employs `scikit-learn` to convert all extracted sections and the persona profile into TF-IDF vectors, representing their semantic importance[cite: 1, 2].
-   **Cosine Similarity Ranking**: Ranks each section by calculating the cosine similarity between its vector and the persona's vector, ensuring the most relevant content is prioritized.

---

## Libraries Used

-   [`PyPDF2`](https://pypdf2.readthedocs.io/en/latest/) (for PDF text extraction) 
-   [`nltk`](https://www.nltk.org/) (for sentence tokenization) 
-   [`scikit-learn`](https://scikit-learn.org/stable/) (for TF-IDF and cosine similarity) 
-   [`numpy`](https://numpy.org/) (for numerical operations) 

---

## How to Run (via Docker)

1.  **Build the Docker Image**:

    ```bash
    docker build --platform=linux/amd64 -t doc-intel-system .
    ```

2.  **Prepare Directories and Input**:
    -   Place your `challenge1b_input.json` file inside the `input` directory[cite: 1, 4].
    -   Place the required PDF documents into their respective `Collection */PDFs/` directories[cite: 1, 4]. The script automatically selects the correct collection based on the `challenge_id` in the input JSON.

3.  **Run the Container**:

    This command mounts your local directories into the container to process the files.

    ```bash
    docker run --rm \
      -v $(pwd)/input:/app/input \
      -v $(pwd)/output:/app/output \
      -v "$(pwd)/Collection 1":"/app/Collection 1" \
      -v "$(pwd)/Collection 2":"/app/Collection 2" \
      -v "$(pwd)/Collection 3":"/app/Collection 3" \
      doc-intel-system
    ```

    **Note for Windows users**: Replace `$(pwd)` with the absolute path to your project directory (e.g., `D:/path/to/project`).

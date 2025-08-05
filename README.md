# COROS Heart Rate Analysis

This project analyzes heart rate data from a COROS `.fit` file, calculates statistics (max/avg HR) for custom laps defined in a `.csv` file, and generates a visualization of the activity.

![Example Output](heart_rate_analysis.png)

## Features

- Parses raw data from `.fit` files.
- Calculates max and average heart rate for user-defined laps.
- Generates a plot visualizing the heart rate trace with laps highlighted.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/heart-rate-analysis.git
    cd heart-rate-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Place your `.fit` file and `laps.csv` file in the `data/` directory. Then run the analysis script:

```bash
python src/analyze_activity.py
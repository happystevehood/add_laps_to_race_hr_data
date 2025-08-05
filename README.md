# .Fit Heart Rate Analysis combined with lap times of cvs file.

This project analyzes heart rate data from a `.fit` file, calculates statistics (max/avg HR) for custom laps defined in a `.csv` file, and generates a visualization of the activity.

<img width="5395" height="3274" alt="26-Jul-25_Steve_Cleary_Redline_Fitness_Games_Singles_Advanced_TimeSeries" src="https://github.com/user-attachments/assets/8f2e163d-df97-4d0c-b65b-c06b441f758d" />

<img width="4774" height="2368" alt="26-Jul-25_Steve_Cleary_Redline_Fitness_Games_Singles_Advanced_Zone_Distribution" src="https://github.com/user-attachments/assets/e1e84d47-1bf2-4157-b5d2-2b23725945e1" />

## Features

- Parses raw data from `.fit` files.
- Generates a plot visualizing the heart rate trace with laps highlighted.

The users needs to provide 3 pieces of information
1) `.fit` file from HRM from a race of their choosing.
2) lap times for the race in the following format.  The first three Fieds, Name, Desciption and Date are mandatory, after that its up to you regarding how many laps. The lap times can be duration or cumulative.

3) your preferred HR zones. Again you can have differnt number of zones with different names to the example. Examples available.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/add_laps_to_race_hr_data.git
    cd add_laps_to_race_hr_data
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

For now you need to update the following variable in the script to point to your relevant files.

# Define the names of your subdirectories
input_dir = 'data/Steve'

output_dir = 'data/Steve'

# Define the paths to your data files
hr_zone_file = 'Steve_HR_Zones.csv'

lap_file_to_process  = 'Steve_Redline_2025_duration.csv'

fit_file_path = 'RedlineFitnessGames_20250726115048.fit'

# Run the scripts
```bash
python add_laps_to_race_hr_data.py

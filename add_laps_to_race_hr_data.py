#in my VSCode 
#activate C:\Users\Steph\miniconda3
#python add_laps_to_race_hr_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import timedelta
import fitparse
import os
import textwrap

# Set plotting style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')

print("Libraries imported successfully.")

# --- User Configuration ---

# Define the names of your subdirectories
input_dir = 'data/Steve'
output_dir = 'data/Steve'

# Define the paths to your data files
hr_zone_file = 'Steve_HR_Zones.csv'

lap_file_to_process  = 'Steve_Redline_2025_duration.csv'
#lap_file_to_process  = 'Steve_Redline_2025_cumulative.csv'
#lap_file_to_process  = 'Steve_Redline_2025_cumulative_test.csv'

# This would be the path to your actual .fit file from your watch
fit_file_path = 'RedlineFitnessGames_20250726115048.fit'

# Define the names of your subdirectories
#input_dir = 'data/Donal'
#output_dir = 'data/Donal'

# Donal's files
#hr_zone_file = 'Donal.csv'

#lap_file_to_process  = 'Redline_Singles_Intermediate.csv'
#fit_file_path = 'Redline_Singles_Intermediate.fit'

#lap_file_to_process  = 'Redline_Mens_Doubles.csv'
#fit_file_path = 'Redline_Mens_Doubles.fit'

#lap_file_to_process  = 'Redline_Mixed_Doubles.csv'
#fit_file_path = 'Redline_Mixed_Doubles.fit'

# --- Construct full paths ---
# This uses os.path.join to create correct paths for any operating system
hr_zone_filepath = os.path.join(input_dir, hr_zone_file)
lap_filepath = os.path.join(input_dir, lap_file_to_process)
fit_filepath = os.path.join(input_dir, fit_file_path)

print(f"Input directory set to: '{input_dir}'")
print(f"Output directory set to: '{output_dir}'")


def load_hr_zones(filepath):
    """
    Loads HR zones from a CSV file and processes the HR range strings.
    
    Args:
        filepath (str): The path to the HR zones CSV file.
        
    Returns:
        pandas.DataFrame: A DataFrame with Zone, min_hr, and max_hr.
    """
    try:
        zones_df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: HR Zone file not found at {filepath}")
        return None

    # Process the 'HR Range' column to create min and max HR columns
    zones_df[['min_hr', 'max_hr']] = zones_df['HR Range'].str.extract(r'(\d+)-?(\d+)?').astype(float)
    
    # Handle special cases like '<126' and '166>'
    zones_df.loc[zones_df['HR Range'].str.contains('<'), 'max_hr'] = zones_df['HR Range'].str.extract(r'<(\d+)').astype(float)[0] - 1
    zones_df.loc[zones_df['HR Range'].str.contains('<'), 'min_hr'] = 0
    
    zones_df.loc[zones_df['HR Range'].str.contains('>'), 'min_hr'] = zones_df['HR Range'].str.extract(r'(\d+)>').astype(float)[0]
    zones_df.loc[zones_df['HR Range'].str.contains('>'), 'max_hr'] = 300 # A practical upper limit for HR

    zones_df = zones_df.set_index('Zone')
    print("HR Zones loaded and processed:")
    print(zones_df)
    return zones_df

def format_duration(td):
    """
    Custom formats a timedelta object for display.
    - Displays 'mm:ss' if the duration is less than 1 hour.
    - Displays 'h:mm:ss' if the duration is 1 hour or more.
    - Handles '0 days' and potential microseconds correctly.
    """
    if pd.isna(td):
        return "" # Return an empty string for any missing or invalid times
    
    # Convert timedelta to a basic string (e.g., '0 days 00:42:14.123')
    # and remove microseconds and the '0 days ' part.
    time_str = str(td).split('.')[0].replace('0 days ', '')
    
    # If the string starts with '00:', it means the duration is less than one hour.
    if time_str.startswith('00:'):
        # Return only the 'mm:ss' part
        return time_str[3:]
    
    # Otherwise, return the full 'h:mm:ss' string
    return time_str

def process_lap_file(filepath):
    """
    Loads lap data from a CSV file, automatically detecting format and
    filtering out empty or invalid columns.
    
    Args:
        filepath (str): The path to the lap data CSV file.
        
    Returns:
        tuple: A tuple containing (laps_df, metadata_dict).
    """
    try:
        df = pd.read_csv(filepath).dropna(how='all').iloc[[0]]
    except FileNotFoundError:
        print(f"Error: Lap file not found at {filepath}")
        return None, None
        
    def _parse_time(s):
        s = str(s).strip().replace('*', '')
        parts = s.split(':')
        seconds = 0
        try:
            if len(parts) == 3:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                seconds = int(parts[0]) * 60 + int(parts[1])
            return pd.to_timedelta(seconds, unit='s')
        except (ValueError, TypeError):
            return pd.NaT

    metadata = {
        'Name': df['Name'].iloc[0],
        'Description': df['Description'].iloc[0],
        'Date': df['Date'].iloc[0]
    }
    
    ## UPDATED ## Robustly filter for valid lap columns only
    potential_lap_columns = [col for col in df.columns if col not in ['Name', 'Description', 'Date']]
    
    lap_columns = []
    lap_timedeltas = []

    for col_name in potential_lap_columns:
        # Rule 1: Skip automatically named empty columns from pandas
        if 'Unnamed' in str(col_name):
            continue
        
        # Rule 2: Try to parse the time value from the cell
        time_val = _parse_time(df[col_name].iloc[0])

        # Rule 3: Only keep this lap if the time value is valid (not NaT)
        if pd.notna(time_val):
            lap_columns.append(col_name)
            lap_timedeltas.append(time_val)
            
    if not lap_columns:
        print("Error: No valid lap data found in the file.")
        return None, None
            
    # --- Auto-detection Logic (now on clean data) ---
    is_cumulative = all(lap_timedeltas[i] <= lap_timedeltas[i+1] for i in range(len(lap_timedeltas) - 1))
    file_type = 'cumulative' if is_cumulative else 'duration'
    print(f"Detected '{file_type.capitalize()}' lap file type.")
    
    # --- Process based on detected file type ---
    laps = []
    total_workout_time = timedelta(0)

    if file_type == 'duration':
        current_time = timedelta(0)
        for i, lap_name in enumerate(lap_columns):
            lap_duration = lap_timedeltas[i]
            start_time = current_time
            end_time = current_time + lap_duration
            laps.append({'Lap Name': lap_name, 'Start Time': start_time, 'End Time': end_time, 'Duration': lap_duration})
            current_time = end_time
        total_workout_time = sum(lap_timedeltas, timedelta())
            
    elif file_type == 'cumulative':
        previous_lap_end_time = timedelta(0)
        for i, lap_name in enumerate(lap_columns):
            end_time = lap_timedeltas[i]
            start_time = previous_lap_end_time
            lap_duration = end_time - start_time
            laps.append({'Lap Name': lap_name, 'Start Time': start_time, 'End Time': end_time, 'Duration': lap_duration})
            previous_lap_end_time = end_time
        total_workout_time = lap_timedeltas[-1] if lap_timedeltas else timedelta(0)

    metadata['Total Time'] = format_duration(total_workout_time)
            
    laps_df = pd.DataFrame(laps)
    print("Lap data processed successfully:")
    print(laps_df)
    return laps_df, metadata

def simulate_fit_data(laps_df, total_duration_seconds):
    """
    !! SIMULATOR FUNCTION !!
    Generates a realistic-looking HR DataFrame since we don't have a .fit file.
    Replace this with parse_fit_file in a real scenario.
    """
    print("\n--- SIMULATING FIT FILE DATA ---")
    timestamps = pd.to_timedelta(np.arange(total_duration_seconds), unit='s')
    hr_data = pd.DataFrame(index=timestamps)
    hr_data['heart_rate'] = 0

    # Base HR for different types of activities
    activity_hr_map = {
        'RUN': 160, 'SKI': 155, 'DEADBALL BURPEES': 165, 'BIKE': 145,
        'FARMER\'S CARRY': 140, 'SHUTTLE RUNS': 170, 'RUSSIAN TWISTS': 130,
        'SANDBAG GAUNTLET': 160, 'ROW': 150, 'SQUAT THRUSTS': 168,
        'THE MULE': 148, 'SLED PUSH & PULL': 175
    }
    
    for _, lap in laps_df.iterrows():
        start_sec = int(lap['Start Time'].total_seconds())
        end_sec = int(lap['End Time'].total_seconds())
        base_hr = activity_hr_map.get(lap['Lap Name'], 140) # Default HR if name not in map
        
        # Create a plausible HR curve for the lap (e.g., rises, stabilizes, falls slightly)
        lap_len = end_sec - start_sec
        ramp_up = int(lap_len * 0.2)
        ramp_down = int(lap_len * 0.1)
        stable = lap_len - ramp_up - ramp_down
        
        hr_profile = np.concatenate([
            np.linspace(base_hr - 15, base_hr + 5, ramp_up),
            np.full(stable, base_hr + 5) + np.random.randn(stable) * 2, # Add noise
            np.linspace(base_hr + 3, base_hr - 10, ramp_down)
        ])
        
        # Ensure array length matches slice
        if lap_len > 0:
            hr_profile = np.resize(hr_profile, lap_len)
            hr_data.iloc[start_sec:end_sec, 0] = hr_profile.astype(int)

    # Fill any gaps (e.g., transition times)
    # FIX: Replaced deprecated .fillna(method='bfill') with .bfill()
    hr_data['heart_rate'] = hr_data['heart_rate'].replace(0, np.nan).interpolate().bfill()
    
    print("Simulated HR data created.")
    return hr_data


def parse_fit_file(filepath):
    """
    Parses a .fit file to extract heart rate data over time.
    
    Args:
        filepath (str): The path to the .fit file.
        
    Returns:
        pandas.DataFrame: DataFrame with a timedelta index and 'heart_rate' column.
    """
    try:
        fitfile = fitparse.FitFile(filepath)
    except FileNotFoundError:
        print(f"Error: FIT file not found at {filepath}")
        return None

    records = []
    for record in fitfile.get_messages('record'):
        # FIX: The 'record' object does not have .get_field(). 
        # We use .get_value() which returns the value or None if the field doesn't exist.
        # This works perfectly for checking and getting the value in one go.
        timestamp = record.get_value('timestamp')
        heart_rate = record.get_value('heart_rate')
        
        # We only proceed if both timestamp and heart_rate are found in the record
        if timestamp is not None and heart_rate is not None:
            records.append({'timestamp': timestamp, 'heart_rate': heart_rate})
    
    if not records:
        print("No heart rate records found in the FIT file.")
        return None

    hr_df = pd.DataFrame(records)
    # Convert absolute timestamps to elapsed time (timedelta)
    hr_df['time'] = pd.to_timedelta(hr_df['timestamp'] - hr_df['timestamp'].iloc[0])
    hr_df = hr_df.set_index('time')[['heart_rate']]
    
    print(f"Successfully parsed .fit file. Found {len(hr_df)} HR records.")
    return hr_df

def read_fit_summary(filepath):
    """
    Reads the pre-calculated summary data from the 'session' message in a FIT file.
    
    Args:
        filepath (str): The path to the .fit file.
        
    Returns:
        dict: A dictionary containing the summary stats, or None if not found.
    """
    try:
        fitfile = fitparse.FitFile(filepath)
    except FileNotFoundError:
        print(f"Error: FIT file not found at {filepath}")
        return None
        
    # Get the 'session' message which contains overall summaries
    for record in fitfile.get_messages('session'):
        summary = {}
        # Iterate over all fields in the message and store them
        for field in record:
            # We only care about fields with values, not units
            if field.units:
                summary[field.name] = f"{field.value} {field.units}"
            else:
                 summary[field.name] = field.value
        
        # A quick check to see if we found useful data
        if 'total_timer_time' in summary:
            print("\nSuccessfully read device summary data from FIT file.")
            return summary
            
    print("\nWarning: Could not find a 'session' summary message in the FIT file.")
    return None


def analyze_activity(hr_df, laps_df, zones_df):
    """
    Analyzes HR data for each lap and overall, calculating stats and time in zones.
    
    Args:
        hr_df (DataFrame): DataFrame of heart rate data.
        laps_df (DataFrame): DataFrame of lap definitions.
        zones_df (DataFrame): DataFrame of HR zone definitions.
        
    Returns:
        DataFrame: A detailed analysis summary for each lap and overall.
    """
    analysis_results = []

    # Bin HR data into zones once for efficiency
    zone_bins = sorted([z['min_hr'] for _, z in zones_df.iterrows()] + [zones_df['max_hr'].max()])
    zone_labels = zones_df.index.tolist()
    hr_df['zone'] = pd.cut(hr_df['heart_rate'], bins=zone_bins, labels=zone_labels, right=False)

    # Time per data point (assuming 1-second recordings)
    time_per_sample = pd.to_timedelta(hr_df.index.to_series().diff().median())
    if pd.isna(time_per_sample):
        time_per_sample = timedelta(seconds=1) # Default if calculation fails
        
    # Analyze each lap
    for _, lap in laps_df.iterrows():
        lap_hr = hr_df[(hr_df.index >= lap['Start Time']) & (hr_df.index < lap['End Time'])]
        
        if lap_hr.empty:
            continue
            
        # Calculate time in each zone for the lap
        time_in_zones = lap_hr['zone'].value_counts() * time_per_sample
        
        lap_stats = {
            'Lap Name': lap['Lap Name'],
            'Duration': lap['Duration'],
            'Avg HR': round(lap_hr['heart_rate'].mean(), 1),
            'Min HR': int(lap_hr['heart_rate'].min()),
            'Max HR': int(lap_hr['heart_rate'].max())
        }
        # Add time-in-zone data to the stats dictionary
        for zone_name in zones_df.index:
            lap_stats[f'Time in {zone_name}'] = time_in_zones.get(zone_name, timedelta(0))
            
        analysis_results.append(lap_stats)

    # --- Add Overall Summary ---
    overall_time_in_zones = hr_df['zone'].value_counts() * time_per_sample
    overall_stats = {
        'Lap Name': 'OVERALL',
        'Duration': laps_df['End Time'].iloc[-1],
        'Avg HR': round(hr_df['heart_rate'].mean(), 1),
        'Min HR': int(hr_df['heart_rate'].min()),
        'Max HR': int(hr_df['heart_rate'].max())
    }
    for zone_name in zones_df.index:
        overall_stats[f'Time in {zone_name}'] = overall_time_in_zones.get(zone_name, timedelta(0))
    analysis_results.append(overall_stats)

    return pd.DataFrame(analysis_results).set_index('Lap Name')

def plot_activity_analysis(hr_df, laps_df, zones_df, metadata, summary_df, smoothing=True, smoothing_window=5, output_path=None):
    """
    Generates a comprehensive plot of the activity analysis, including a summary table,
    and optionally saves the plot to a file.
    
    Args:
        ... (all previous args) ...
        output_path (str, optional): The full path to save the output PNG file. Defaults to None.
    """
    fig, axs = plt.subplots(
        2, 1, 
        figsize=(18, 11),
        gridspec_kw={'height_ratios': [3, 1]}
    )
    
    # --- Plot 1: The Heart Rate Graph ---
    graph_ax = axs[0]
    graph_ax.set_ylim(bottom=hr_df['heart_rate'].min() - 10, top=hr_df['heart_rate'].max() + 15)
    
    plot_zones_df = zones_df.copy()
    plot_top_y = graph_ax.get_ylim()[1]
    plot_zones_df.iloc[-1, plot_zones_df.columns.get_loc('max_hr')] = plot_top_y

    cmap = plt.get_cmap('RdYlGn_r')
    num_zones = len(plot_zones_df)
    colors = [cmap(i) for i in np.linspace(0, 1, num_zones)]
    zone_color_map = {zone: colors[i] for i, zone in enumerate(plot_zones_df.index)}

    for zone, z_data in plot_zones_df.iterrows():
        graph_ax.axhspan(z_data['min_hr'], z_data['max_hr'], 
                         color=zone_color_map[zone], alpha=0.3, 
                         label=f"{zone} ({z_data['HR Range']})")

    time_in_minutes = hr_df.index.total_seconds() / 60
    
    if smoothing:
        hr_to_plot = hr_df['heart_rate'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        hr_line_label = 'Heart Rate (BPM, Smoothed)'
    else:
        hr_to_plot = hr_df['heart_rate']
        hr_line_label = 'Heart Rate (BPM, Raw)'
        
    graph_ax.plot(time_in_minutes, hr_to_plot, color='black', linewidth=1.5, label=hr_line_label)

    avg_hr = summary_df.loc['OVERALL', 'Avg HR']
    graph_ax.axhline(y=avg_hr, color='dodgerblue', linestyle='--', linewidth=1.5, label=f'Avg HR ({avg_hr:.0f})')
    graph_ax.text(graph_ax.get_xlim()[1] * 0.98, avg_hr, f' Avg: {avg_hr:.0f} ', 
                  color='white', backgroundcolor='dodgerblue', ha='right', va='center', fontsize=8)

    max_hr_val = hr_df['heart_rate'].max()
    max_hr_time_min = (hr_df['heart_rate'].idxmax()).total_seconds() / 60
    graph_ax.plot(max_hr_time_min, max_hr_val, marker='*', markersize=15, 
                  color='gold', markeredgecolor='black', linestyle='none', label=f'Max HR ({max_hr_val})')
    graph_ax.text(max_hr_time_min, max_hr_val + 1, f' {max_hr_val}',
                  ha='left', va='bottom', fontsize=9, fontweight='bold')

    for _, lap in laps_df.iterrows():
        lap_end_min = lap['End Time'].total_seconds() / 60
        graph_ax.axvline(x=lap_end_min, color='dimgray', linestyle='--', linewidth=1.2)
        
        lap_center_min = (lap['Start Time'].total_seconds() + lap['End Time'].total_seconds()) / 120
        wrapped_label = textwrap.fill(lap['Lap Name'], width=10)
        graph_ax.text(lap_center_min, -0.02, wrapped_label, rotation=90, ha='center', va='top', 
                      fontsize=8, style='italic', color='black', transform=graph_ax.get_xaxis_transform())

    graph_ax.set_title(f"Heart Rate Analysis: {metadata['Name']} - {metadata['Description']}\nDate: {metadata['Date']} | Total Time: {metadata['Total Time']}", fontsize=16, fontweight='bold')
    graph_ax.set_ylabel("Heart Rate (BPM)", fontsize=12)
    graph_ax.tick_params(axis='x', labelbottom=False)
    graph_ax.set_xlim(0, laps_df['End Time'].iloc[-1].total_seconds() / 60)

    handles, labels = graph_ax.get_legend_handles_labels()
    label_handle_map = {label: handle for handle, label in zip(handles, labels)}

    ordered_labels = []
    if hr_line_label in label_handle_map: ordered_labels.append(hr_line_label)
    if f'Avg HR ({avg_hr:.0f})' in label_handle_map: ordered_labels.append(f'Avg HR ({avg_hr:.0f})')
    if f'Max HR ({max_hr_val})' in label_handle_map: ordered_labels.append(f'Max HR ({max_hr_val})')
    for zone in zones_df.index[::-1]:
        for label in labels:
            if label.startswith(zone): ordered_labels.append(label); break
    
    final_handles = [label_handle_map[lbl] for lbl in ordered_labels]
    graph_ax.legend(final_handles, ordered_labels, loc='lower right', fontsize=9, ncol=1)

    table_ax = axs[1]
    table_ax.axis('off')
    table_rows = ['Duration', 'Avg HR', 'Min HR', 'Max HR']
    y_positions = [0.9, 0.75, 0.6, 0.45]
    table_ax.set_xlim(graph_ax.get_xlim())

    for i, row_title in enumerate(table_rows):
        table_ax.text(0, y_positions[i], f"{row_title}:", ha='right', va='center', fontweight='bold', fontsize=10)

    for _, lap in laps_df.iterrows():
        lap_center_min = (lap['Start Time'].total_seconds() + lap['End Time'].total_seconds()) / 120
        lap_stats = summary_df.loc[lap['Lap Name']]
        for i, stat_name in enumerate(table_rows):
            if stat_name in lap_stats:
                table_ax.text(lap_center_min, y_positions[i], lap_stats[stat_name], ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    
    # --- ## NEW: Save the figure before showing it ## ---
    if output_path:
        try:
            # bbox_inches='tight' ensures the saved image includes all labels without being cut off
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nGraph successfully saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving graph: {e}")
            
    plt.show()
    
def plot_zone_distribution_chart(summary_df, zones_df, metadata, output_path=None):
    """
    Generates a 100% stacked bar chart showing the percentage of time
    spent in each HR zone for each lap.
    
    Args:
        summary_df (DataFrame): The analysis summary DataFrame.
        zones_df (DataFrame): The HR zone definitions.
        metadata (dict): Dictionary of activity metadata.
        output_path (str, optional): The full path to save the output PNG file.
    """
    # 1. Prepare the data for plotting
    laps_summary = summary_df.drop('OVERALL', errors='ignore')
    zone_cols = [f'Time in {zone}' for zone in zones_df.index]
    zone_times_sec = laps_summary[zone_cols].apply(lambda x: x.dt.total_seconds())
    
    zone_percentages = zone_times_sec.div(zone_times_sec.sum(axis=1), axis=0) * 100
    zone_percentages.fillna(0, inplace=True)
    zone_percentages.columns = zones_df.index
    
    # 2. Create the color map
    cmap = plt.get_cmap('RdYlGn_r')
    num_zones = len(zones_df)
    colors = [cmap(i) for i in np.linspace(0, 1, num_zones)]
    
    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # ## UPDATED ## Plot the stacked bar chart in the natural order
    # This plots 'Recovery' (green) at the bottom and 'Anaerobic Power' (red) at the top.
    zone_percentages.plot(
        kind='bar', 
        stacked=True, 
        color=colors, # Use the natural color order
        ax=ax,
        width=0.8
    )
    
    # 4. Format the plot for readability
    ax.set_title(f"HR Zone Distribution per Lap: {metadata['Name']} - {metadata['Description']}", fontsize=16, fontweight='bold')
    ax.set_ylabel("Percentage of Time (%)", fontsize=12)
    ax.set_xlabel("Workout Lap", fontsize=12)
    
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 100)
    
    plt.xticks(rotation=45, ha='right')
    
    ax.legend(title='HR Zones', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center', color='black', fontsize=9,
                     padding=-10,
                     labels=[f'{w:.0f}%' if w > 5 else '' for w in container.datavalues])

    plt.tight_layout()
    
    # 5. Save the figure if an output path is provided
    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nZone distribution chart successfully saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving zone distribution chart: {e}")
            
    plt.show()

# --- Main Script Execution ---
def main():
    # 1. Load HR Zones
    hr_zones = load_hr_zones(hr_zone_filepath)

    # 2. Load Lap Data
    laps_info, activity_metadata = process_lap_file(lap_filepath)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if hr_zones is not None and laps_info is not None:
        # 3. Get HR Data from the FIT file
        hr_timeseries = parse_fit_file(fit_filepath)

        if hr_timeseries is not None:
            # 4. Perform the detailed analysis on the raw time-series data
            analysis_summary = analyze_activity(hr_timeseries, laps_info, hr_zones)
            
            display_summary = analysis_summary.copy()
            display_summary['Duration'] = display_summary['Duration'].apply(format_duration)
            for col in display_summary.columns:
                if 'Time in' in str(col):
                    # We need the original timedelta objects for the bar chart, so we don't format here
                    pass

            # 5. Display the calculated analysis table
            print("\n" + "="*70)
            print("          PYTHON-CALCULATED ANALYSIS (from raw second-by-second data)")
            print("="*70)
            # We'll display a version with formatted timedeltas for readability
            formatted_display = display_summary.copy()
            for col in formatted_display.columns:
                if isinstance(formatted_display[col].iloc[0], timedelta):
                    formatted_display[col] = formatted_display[col].apply(lambda x: str(x).split('.')[0].replace('0 days ', ''))
            print(formatted_display)
            
            # 6. Read and display the device's own summary for comparison
            device_summary = read_fit_summary(fit_filepath)
            if device_summary:
                print("\n" + "="*70)
                print("            DEVICE SUMMARY (read directly from FIT file session)")
                print("="*70)
                print(f"  - Avg HR: {device_summary.get('avg_heart_rate', 'N/A')}")
                print(f"  - Max HR: {device_summary.get('max_heart_rate', 'N/A')}")
                print(f"  - Min HR: {device_summary.get('min_heart_rate', 'N/A')}")
                print(f"  - Total Time: {device_summary.get('total_timer_time', 'N/A')}")
                print(f"  - Calories: {device_summary.get('total_calories', 'N/A')}")
                print(f"  - Distance: {device_summary.get('total_distance', 'N/A')}")
            
            # --- 7. Generate and save the plots ---
            
            # Create a base filename for all outputs for this activity
            name = activity_metadata['Name'].replace(' ', '_')
            desc = activity_metadata['Description'].replace(' ', '_').replace('/', '-')
            date = activity_metadata['Date']
            base_filename = f"{date}_{name}_{desc}"
            
            # -- Plot 1: Time-Series Graph --
            output_filepath_timeseries = os.path.join(output_dir, f"{base_filename}_TimeSeries.png")
            plot_activity_analysis(
                hr_timeseries, laps_info, hr_zones, activity_metadata, display_summary,
                output_path=output_filepath_timeseries
            )

            # -- ## NEW: Plot 2: Zone Distribution Bar Chart ## --
            output_filepath_barchart = os.path.join(output_dir, f"{base_filename}_Zone_Distribution.png")
            # Note: We pass the ORIGINAL analysis_summary with timedelta objects here
            plot_zone_distribution_chart(
                analysis_summary, hr_zones, activity_metadata,
                output_path=output_filepath_barchart
            )
            
if __name__ == "__main__":
    main()
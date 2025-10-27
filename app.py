import os
import io
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
import logging

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
NO_COLS = [f'No{i}' for i in range(1, 28)]

# --- Helper Functions (as per request) ---

def parse_date_dayfirst(s: pd.Series) -> pd.Series:
    """
    Parses a pd.Series into datetime objects robustly, prioritizing day-first format.
    Handles Excel serial dates, common day-first string formats, and a general fallback.
    """
    # Convert NaNs, empty strings, 'NaT' to actual NaNs for consistent handling
    s_cleaned = s.replace({None: np.nan, 'NaT': np.nan, '': np.nan}).astype(str).replace('nan', np.nan)

    # Step 1: Excel Serial Dates
    excel_dates = pd.to_numeric(s_cleaned, errors='coerce')
    is_excel_date = (excel_dates >= 1) & (excel_dates <= 80000) & (~excel_dates.isna())
    
    parsed_dates = pd.NaT.to_series(index=s.index)
    if is_excel_date.any():
        excel_epoch = datetime(1899, 12, 30) # Excel's 1900-based date system origin
        # pd.to_datetime with unit='D' and origin handles the conversion from serial number
        parsed_dates.loc[is_excel_date] = pd.to_datetime(excel_dates.loc[is_excel_date], unit='D', origin=excel_epoch)

    # Step 2: Common Day-First String Formats
    remaining_indices = parsed_dates.isna() & (~s_cleaned.isna())
    if remaining_indices.any():
        common_dayfirst_formats = [
            '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y',
            '%d-%m-%y', '%d/%m/%y',
            '%Y-%m-%d' # Also include YYYY-MM-DD as it's common and can be day-first if ambiguous
        ]
        for fmt in common_dayfirst_formats:
            try:
                temp_parsed = pd.to_datetime(s_cleaned[remaining_indices], format=fmt, errors='coerce')
                parsed_dates.loc[remaining_indices & ~temp_parsed.isna()] = temp_parsed.loc[remaining_indices & ~temp_parsed.isna()]
                remaining_indices = parsed_dates.isna() & (~s_cleaned.isna())
                if not remaining_indices.any(): # Stop if all remaining values are parsed
                    break
            except Exception:
                # Continue to next format if parsing fails for any reason
                continue 

    # Step 3: General Fallback
    remaining_indices = parsed_dates.isna() & (~s_cleaned.isna())
    if remaining_indices.any():
        parsed_dates.loc[remaining_indices] = pd.to_datetime(s_cleaned[remaining_indices], dayfirst=True, errors='coerce')

    return parsed_dates.dt.normalize() # Remove time component


def load_csv(path: Path) -> pd.DataFrame:
    """
    Loads the input CSV, cleans data, normalizes column names, detects and standardizes
    the date column, and ensures the existence of No1 to No27 numeric columns.
    """
    if not path.exists():
        app.logger.error(f"Input file not found: {path}")
        raise FileNotFoundError(f"Input file not found: {path}")

    # Read CSV with all columns as string to prevent premature type conversion
    # and keep empty strings explicitly.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8')

    # Clean column names (remove BOM, strip whitespace)
    df.columns = [col.replace('\ufeff', '').strip() for col in df.columns]

    # Handle sub-header row (if "STT" column exists and has specific values)
    # Interpretation: If the 'STT' column contains specific string values, drop those rows.
    # This assumes the true header is correctly parsed by pd.read_csv (header=0).
    if 'STT' in df.columns:
        initial_rows_count = df.shape[0]
        df = df[~df['STT'].isin(['STT', 'No.', 'No'])].reset_index(drop=True)
        if df.shape[0] < initial_rows_count:
            app.logger.info(f"Dropped {initial_rows_count - df.shape[0]} sub-header like rows based on 'STT' column values.")

    # Detect and standardize date column
    date_col_name = None
    if 'Date' in df.columns:
        date_col_name = 'Date'
        app.logger.info("Using existing 'Date' column.")
    else:
        # Try to find the best candidate for date column
        best_date_col = None
        max_valid_dates = -1
        for col in df.columns:
            # Check for a reasonable number of unique values that aren't empty
            if df[col].nunique(dropna=True) < 2: # A date column should have some variation
                continue
            
            # Attempt to parse as date and count valid dates
            parsed = parse_date_dayfirst(df[col])
            valid_dates_count = parsed.count() # Number of non-NaT values
            
            # Candidate column must have at least 50% valid dates and more valid dates than previous best
            if valid_dates_count > max_valid_dates and valid_dates_count >= (len(df) * 0.5):
                max_valid_dates = valid_dates_count
                best_date_col = col
        
        if best_date_col:
            app.logger.info(f"Detected '{best_date_col}' as the date column based on validity.")
            df['Date'] = parse_date_dayfirst(df[best_date_col])
            # If the best_date_col is not 'Date', drop the original to avoid redundancy
            if best_date_col != 'Date':
                df = df.drop(columns=[best_date_col])
            date_col_name = 'Date'
        else:
            app.logger.error("No suitable date column found in the CSV based on content analysis.")
            raise ValueError("No suitable date column found in the CSV. Ensure there is a 'Date' column or a column with clear date values.")
    
    # Ensure 'Date' column is in datetime format after potential detection/re-parsing
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = parse_date_dayfirst(df['Date'])
    elif 'Date' not in df.columns: # This case should ideally be caught by previous logic
        raise ValueError("Date column could not be established in the DataFrame.")


    # Ensure numeric columns `No1` to `No27`
    for col in NO_COLS:
        if col not in df.columns:
            df[col] = '' # Add empty string column if missing
            app.logger.warning(f"Column '{col}' was missing and added as empty.")

    # Drop rows where 'Date' value is NaT (invalid dates), sort, and reset index
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    if df.empty:
        raise ValueError("DataFrame is empty after date cleaning. Check your input CSV or date parsing logic.")

    return df[['Date'] + NO_COLS]


def count_digits_0_9(row: pd.Series) -> dict:
    """
    Counts the frequency of each digit (0-9) in the tens and units place
    of numbers found in columns No1 to No27 for a given row.
    """
    counts = {str(i): 0 for i in range(10)}
    
    for col in NO_COLS:
        value = str(row[col]).strip()
        if not value: # Skip empty values
            continue

        digits_to_count = []
        try:
            # Try converting to integer
            num = int(float(value)) # Use float first to handle potential "1.0" or similar
            s_num = f"{abs(num):02d}" # Format absolute value as two digits (e.g., 5 -> "05", -5 -> "05")
            
            # Count tens and units place
            if len(s_num) >= 2:
                digits_to_count.append(s_num[-2]) # Tens place
                digits_to_count.append(s_num[-1]) # Units place
            elif len(s_num) == 1: # Single digit numbers only count units place (implicitly tens is 0)
                digits_to_count.append('0') # Tens place is 0
                digits_to_count.append(s_num[-1]) # Units place
        except ValueError:
            # If not a clean integer, extract all digits from the string
            all_digits = [c for c in value if c.isdigit()]
            if len(all_digits) >= 2:
                digits_to_count.append(all_digits[-2]) # Tens place (last two digits of all extracted digits)
                digits_to_count.append(all_digits[-1]) # Units place
            elif len(all_digits) == 1:
                digits_to_count.append('0') # Tens place is 0
                digits_to_count.append(all_digits[-1])

        for digit in digits_to_count:
            if '0' <= digit <= '9':
                counts[digit] += 1
    return counts


def make_rows_for_date(date_val: pd.Timestamp, counts: dict) -> list:
    """
    Creates 10 output data rows (Min1-Min5, Max1-Max5) for a specific date,
    based on the counted digit frequencies.
    """
    # Group digits by their frequency
    # freq -> list of digits (sorted for consistent output)
    freq_map = {} 
    for digit, freq in counts.items():
        if freq not in freq_map:
            freq_map[freq] = []
        freq_map[freq].append(digit)

    # Get unique frequencies and sort them for Min and Max
    unique_freqs_sorted_asc = sorted(list(freq_map.keys()))
    unique_freqs_sorted_desc = sorted(list(freq_map.keys()), reverse=True)
    
    out_rows = []

    def create_output_row_dict(label: str, freq_val: str, digits_list: list) -> dict:
        row_dict = {
            "Date": date_val,
            "D2": label,
            "Freq": freq_val,
            "Count": len(digits_list) if freq_val != '' else ''
        }
        for d in range(10):
            row_dict[str(d)] = str(d) if str(d) in digits_list else ''
        return row_dict

    # Min1-Min5
    for i in range(5):
        label = f"Min{i+1}"
        if i < len(unique_freqs_sorted_asc):
            freq = unique_freqs_sorted_asc[i]
            digits = sorted(freq_map[freq]) # Sort digits within the same frequency for consistency
            out_rows.append(create_output_row_dict(label, freq, digits))
        else:
            # Fill with empty data if less than 5 unique frequencies
            out_rows.append(create_output_row_dict(label, '', []))

    # Max1-Max5
    for i in range(5):
        label = f"Max{i+1}"
        if i < len(unique_freqs_sorted_desc):
            freq = unique_freqs_sorted_desc[i]
            digits = sorted(freq_map[freq]) # Sort digits within the same frequency for consistency
            out_rows.append(create_output_row_dict(label, freq, digits))
        else:
            # Fill with empty data if less than 5 unique frequencies
            out_rows.append(create_output_row_dict(label, '', []))
            
    return out_rows


@app.route('/', methods=['GET'])
def home():
    """Simple health check endpoint."""
    app.logger.info("Home endpoint accessed.")
    return jsonify({"message": "Python Flask backend is running and ready to process CSV files."})


@app.route('/process_csv', methods=['POST'])
def process_csv_endpoint():
    """
    API endpoint to receive a CSV file, process it, and return the result as a new CSV.
    """
    if 'file' not in request.files:
        app.logger.warning("No file part in the request.")
        return jsonify({"error": "No file part in the request. Please upload a CSV file."}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.warning("No selected file.")
        return jsonify({"error": "No selected file. Please choose a CSV file."}), 400

    if file and file.filename.endswith('.csv'):
        try:
            # Create a temporary directory to store input and output files
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                input_file_path = tmp_path / "Lucky.csv"
                
                # Save the uploaded file to the temporary path
                file.save(input_file_path)
                app.logger.info(f"Uploaded file saved to: {input_file_path}")

                # Load and process the input CSV
                input_df = load_csv(input_file_path)
                app.logger.info(f"Input CSV loaded with {len(input_df)} rows after cleaning.")

                out_rows = []
                for index, row in input_df.iterrows():
                    date_val = row['Date']
                    digit_counts = count_digits_0_9(row)
                    out_rows.extend(make_rows_for_date(date_val, digit_counts))
                
                app.logger.info(f"Processed data for {len(input_df)} dates, generated {len(out_rows)} output rows.")

                # Create the output DataFrame
                output_columns = ["Date", "D2", "Freq", "Count"] + [str(d) for d in range(10)]
                output_df = pd.DataFrame(out_rows, columns=output_columns)

                # Format Date column for output
                output_df['Date'] = output_df['Date'].dt.strftime("%d-%m-%Y")

                # Save the output to a temporary CSV file
                output_file_name = "D2.csv"
                output_file_path = tmp_path / output_file_name
                output_df.to_csv(output_file_path, index=False, encoding='utf-8')
                app.logger.info(f"Output CSV saved to: {output_file_path}")

                # Send the processed CSV file back as a response
                return send_file(
                    output_file_path,
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=output_file_name
                )

        except FileNotFoundError as e:
            app.logger.error(f"File not found error: {e}")
            return jsonify({"error": str(e)}), 404
        except ValueError as e:
            app.logger.error(f"Data processing error: {e}")
            return jsonify({"error": f"Error processing CSV data: {str(e)}"}), 400
        except Exception as e:
            app.logger.error(f"An unexpected error occurred during CSV processing: {e}", exc_info=True)
            return jsonify({"error": f"An internal server error occurred: {str(e)}. Please check server logs for details."}), 500
    else:
        app.logger.warning(f"Invalid file type uploaded: {file.filename if file else 'None'}")
        return jsonify({"error": "Invalid file type. Please upload a .csv file."}), 400

if __name__ == '__main__':
    # When running locally, Flask's development server is used
    # On Render, gunicorn will be used, which handles port automatically
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
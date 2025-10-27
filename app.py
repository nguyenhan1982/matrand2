import os
import io
import uuid
import pandas as pd
import dateparser
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def find_and_standardize_date_column(df):
    best_date_col = None
    max_valid_dates = 0
    df_copy = df.copy()

    for col in df_copy.columns:
        # Try to parse dates robustly
        parsed_dates = df_copy[col].apply(lambda x: dateparser.parse(str(x)) if pd.notna(x) else None)
        valid_dates_count = parsed_dates.count()

        if valid_dates_count > max_valid_dates:
            max_valid_dates = valid_dates_count
            best_date_col = col
            df_copy['StandardDate'] = parsed_dates # Store parsed dates temporarily

    if best_date_col:
        # Format the standard date column
        df_copy['StandardDate'] = df_copy['StandardDate'].dt.strftime('%d-%m-%Y')
        # Remove original date column if it's not needed or rename 'StandardDate'
        # For simplicity, we'll keep 'StandardDate' and drop the original best_date_col if it's different.
        if best_date_col != 'StandardDate' and 'StandardDate' in df_copy.columns:
             if best_date_col in df_copy.columns: # Check if column still exists
                 df_copy = df_copy.drop(columns=[best_date_col])
             df_copy = df_copy.rename(columns={'StandardDate': 'Ngay'})
        else: # If the best date column was already named 'StandardDate' (unlikely) or if we just want to replace
             df_copy = df_copy.rename(columns={'StandardDate': 'Ngay'})
    else:
        raise ValueError("Could not find a suitable date column in the uploaded data.")

    # Reorder columns to have 'Ngay' first
    cols = ['Ngay'] + [col for col in df_copy.columns if col != 'Ngay']
    return df_copy[cols]


def process_lottery_data(filepath):
    df = pd.read_csv(filepath)

    # 1. Read and Standardize Date Column
    df_processed = find_and_standardize_date_column(df)

    summary_records = []

    # Get all columns that are not the 'Ngay' column
    number_cols = [col for col in df_processed.columns if col != 'Ngay']

    for index, row in df_processed.iterrows():
        current_date_str = row['Ngay']
        digit_counts = {str(i): 0 for i in range(10)}

        # 2. Analyze Digits for each day
        all_numbers_for_day = []
        for col in number_cols:
            try:
                # Ensure the value is a string before checking if it's numeric
                val = str(row[col]).strip()
                if val.isdigit(): # Check if it's a positive integer string
                    num = int(val)
                    all_numbers_for_day.append(num)
                elif pd.isna(row[col]): # Handle NaN values
                    continue
                else: # Try converting non-integer strings to float then int, handling potential decimal numbers
                    try:
                        num = int(float(val))
                        all_numbers_for_day.append(num)
                    except ValueError:
                        continue # Skip values that cannot be converted to numbers
            except (ValueError, TypeError):
                continue # Skip values that cannot be converted to numbers

        for num in all_numbers_for_day:
            last_two_digits_str = str(num % 100).zfill(2) # e.g., 5 -> "05", 123 -> "23"
            for digit_char in last_two_digits_str:
                if digit_char in digit_counts:
                    digit_counts[digit_char] += 1

        # 3. Create Summary Table for each day
        # Get unique frequencies and their associated digits
        freq_map = {} # freq -> list of digits
        for digit, count in digit_counts.items():
            if count not in freq_map:
                freq_map[count] = []
            freq_map[count].append(digit)

        unique_frequencies = sorted(list(freq_map.keys()))

        # 5 Min frequencies
        for i in range(min(5, len(unique_frequencies))):
            freq = unique_frequencies[i]
            digits = sorted(freq_map[freq])
            summary_records.append({
                'Ngay': current_date_str,
                'Loai': f'Min{i+1}',
                'TanSuat': freq,
                'SoLuongChuSo': len(digits),
                'CacChuSo': ', '.join(digits)
            })
        # Fill remaining Min slots if less than 5 unique frequencies
        for i in range(len(unique_frequencies), 5):
            summary_records.append({
                'Ngay': current_date_str,
                'Loai': f'Min{i+1}',
                'TanSuat': '',
                'SoLuongChuSo': '',
                'CacChuSo': ''
            })

        # 5 Max frequencies
        for i in range(min(5, len(unique_frequencies))):
            freq = unique_frequencies[len(unique_frequencies) - 1 - i]
            digits = sorted(freq_map[freq])
            summary_records.append({
                'Ngay': current_date_str,
                'Loai': f'Max{i+1}',
                'TanSuat': freq,
                'SoLuongChuSo': len(digits),
                'CacChuSo': ', '.join(digits)
            })
        # Fill remaining Max slots if less than 5 unique frequencies
        for i in range(len(unique_frequencies), 5):
            summary_records.append({
                'Ngay': current_date_str,
                'Loai': f'Max{i+1}',
                'TanSuat': '',
                'SoLuongChuSo': '',
                'CacChuSo': ''
            })

    output_df = pd.DataFrame(summary_records)
    
    # 4. Save results to a new CSV
    output_filename = f"ket_qua_phan_tich_{uuid.uuid4().hex}.csv"
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    output_df.to_csv(output_filepath, index=False, encoding='utf-8')

    return summary_records, output_filename

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            summary_data, output_filename = process_lottery_data(filepath)

            # Clean up the temporary uploaded file
            os.remove(filepath)

            return jsonify({
                'message': 'File processed successfully',
                'summary_data': summary_data,
                'download_url': f'/download/{output_filename}'
            }), 200
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'The uploaded CSV file is empty.'}), 400
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    return jsonify({'error': 'Something went wrong.'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
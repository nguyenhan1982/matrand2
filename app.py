import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import io
import base64
from collections import defaultdict
import dateutil.parser
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes, allowing frontend from Netlify to connect

def parse_date_robust(date_val):
    """
    Robustly parses a date value, handling various string formats and Excel serial numbers.
    """
    if pd.isna(date_val):
        return pd.NaT

    # 1. Attempt to convert Excel serial number (heuristic: large positive integer/float)
    # Excel date epoch: 1899-12-30 for Windows. Serial 1 is 1900-01-01.
    if isinstance(date_val, (int, float)) and date_val > 1 and date_val < 70000:  # Broader range for Excel serials
        try:
            return pd.to_datetime(date_val, unit='D', origin='1899-12-30')
        except (ValueError, TypeError):
            pass

    # 2. Try pandas to_datetime with infer_datetime_format for common string formats
    try:
        return pd.to_datetime(str(date_val), infer_datetime_format=True)
    except (ValueError, TypeError):
        pass

    # 3. Fallback to dateutil for more flexible parsing if pandas fails
    try:
        return dateutil.parser.parse(str(date_val))
    except (ValueError, TypeError):
        return pd.NaT

@app.route('/process_csv', methods=['POST'])
def process_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file nào được tải lên.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn.'}), 400
    
    if file:
        try:
            # Read CSV content from the uploaded file
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        except Exception as e:
            return jsonify({'error': f'Lỗi khi đọc file CSV: {e}'}), 400

        # 1. Tải và Chuẩn bị Dữ liệu: Parse 'Ngày' and identify number columns
        date_column_name = None
        # Prioritize 'Ngày' then common English date column names
        for col_name_candidate in ['Ngày', 'Ngày Tháng', 'Date', 'DATE']:
            if col_name_candidate in df.columns:
                date_column_name = col_name_candidate
                break
        
        if not date_column_name:
            return jsonify({'error': 'Không tìm thấy cột ngày (ví dụ: "Ngày", "Date") trong file CSV.'}), 400

        # Apply robust date parsing
        df['Parsed_Date'] = df[date_column_name].apply(parse_date_robust)
        
        # Filter out rows where date parsing failed
        df = df.dropna(subset=['Parsed_Date'])
        
        if df.empty:
            return jsonify({'error': 'Không có dữ liệu hợp lệ sau khi phân tích cột ngày.'}), 400

        # Standardize date format for consistent grouping (YYYY-MM-DD)
        df['Formatted_Date'] = df['Parsed_Date'].dt.strftime('%Y-%m-%d')

        # Identify number columns (e.g., 'No1', 'No2', etc. or other numeric-looking columns)
        number_columns = [col for col in df.columns if re.match(r'No\d+', str(col), re.IGNORECASE)]
        
        # Fallback: if no 'NoX' columns, check for other columns that are predominantly numeric
        if not number_columns:
            for col in df.columns:
                # Exclude original date column and the newly created date columns
                if col in [date_column_name, 'Parsed_Date', 'Formatted_Date']:
                    continue
                # Check if values can be converted to numbers (allowing NaNs)
                is_numeric = pd.to_numeric(df[col], errors='coerce').notna().sum()
                # If more than 50% of values are numeric, consider it a number column
                if is_numeric / len(df) > 0.5:
                    number_columns.append(col)

        if not number_columns:
            return jsonify({'error': 'Không tìm thấy cột số (ví dụ: "No1", "No2") trong file CSV.'}), 400

        # 2. Đếm Tần suất Chữ số
        daily_digit_counts = defaultdict(lambda: defaultdict(int)) # {date: {digit: count}}

        for index, row in df.iterrows():
            date_str = row['Formatted_Date']
            for col in number_columns:
                value = row[col]
                
                if pd.isna(value):
                    continue
                
                # Convert value to string and extract only digit characters
                s_value = str(value)
                numeric_part = ''.join(filter(str.isdigit, s_value))
                
                if not numeric_part:
                    continue
                
                # Apply the specific digit counting rule:
                # - If the number has less than two digits, count its single digit.
                # - If the number has two or more digits, count its last two digits.
                if len(numeric_part) < 2:
                    digit = int(numeric_part)
                    daily_digit_counts[date_str][digit] += 1
                else:
                    # Extract last two digits as integers
                    digit1 = int(numeric_part[-2])
                    digit2 = int(numeric_part[-1])
                    daily_digit_counts[date_str][digit1] += 1
                    daily_digit_counts[date_str][digit2] += 1
        
        # 3. Tạo Báo cáo Tần suất
        report_data = []
        report_columns = ['Ngày', 'Loại Báo Cáo', 'Tần suất', 'Số lượng chữ số có tần suất này', 'Các chữ số (0-9)']

        # Sort dates for chronological report order
        for date_str in sorted(daily_digit_counts.keys()):
            digit_counts = daily_digit_counts[date_str]
            
            # Group digits by their frequency
            freq_to_digits = defaultdict(list)
            for digit, count in digit_counts.items():
                freq_to_digits[count].append(digit)
            
            # Get distinct frequencies and sort them
            distinct_frequencies = sorted(list(freq_to_digits.keys()))

            # Tần suất Thấp nhất (5 hàng)
            # Take up to 5 lowest distinct frequencies
            for freq in distinct_frequencies[:5]:
                digits = sorted(freq_to_digits[freq])
                report_data.append([
                    date_str, 
                    'Thấp nhất', 
                    freq, 
                    len(digits), 
                    ','.join(map(str, digits))
                ])

            # Tần suất Cao nhất (5 hàng)
            # Take up to 5 highest distinct frequencies, sorted in descending order
            highest_freqs = sorted(distinct_frequencies, reverse=True)[:5]
            for freq in highest_freqs:
                digits = sorted(freq_to_digits[freq])
                report_data.append([
                    date_str, 
                    'Cao nhất', 
                    freq, 
                    len(digits), 
                    ','.join(map(str, digits))
                ])

        # 4. Xuất Dữ liệu
        if not report_data:
            return jsonify({'error': 'Không có dữ liệu báo cáo được tạo. Kiểm tra định dạng dữ liệu đầu vào.'}), 400

        report_df = pd.DataFrame(report_data, columns=report_columns)
        
        # Convert the report DataFrame to a CSV string in memory
        output_buffer = io.StringIO()
        report_df.to_csv(output_buffer, index=False, encoding='utf-8')
        output_csv_string = output_buffer.getvalue()

        # Base64 encode the CSV string for safe transfer over JSON
        encoded_csv = base64.b64encode(output_csv_string.encode('utf-8')).decode('utf-8')

        return jsonify({'filename': 'bao_cao_tan_suat_chu_so.csv', 'csv_data': encoded_csv})

if __name__ == '__main__':
    # Use environment variable for port for Render deployment, fallback to 5000 for local dev
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
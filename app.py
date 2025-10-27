import pandas as pd
import numpy as np
from pathlib import Path
import re
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
import os
import shutil

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Define constants for numeric columns
NO_COLS = [f'No{i}' for i in range(1, 28)]

def parse_date_dayfirst(s: pd.Series) -> pd.Series:
    """
    Phân tích cú pháp một pd.Series thành các đối tượng datetime64[ns] một cách mạnh mẽ,
    ưu tiên định dạng ngày-đầu-tiên (day-first).
    """
    # Bước 0: Chuyển đổi các giá trị rỗng, NaT, nan, None thành None để xử lý nhất quán
    s = s.replace({np.nan: None, pd.NaT: None, 'nan': None, '': None}).astype(str)

    parsed_dates = pd.Series([pd.NaT] * len(s), index=s.index, dtype='datetime64[ns]')
    
    # Bước 1 (Excel Serial Dates): Cố gắng chuyển đổi các giá trị số thành ngày serial của Excel
    # Chỉ xử lý các giá trị không phải NaT và có thể chuyển đổi thành số
    numeric_s = pd.to_numeric(s, errors='coerce')
    excel_mask = (numeric_s.notna()) & (numeric_s >= 1) & (numeric_s <= 80000)
    
    if excel_mask.any():
        excel_dates = pd.to_datetime(numeric_s[excel_mask], unit='D', origin=pd.Timestamp('1899-12-30'), errors='coerce')
        parsed_dates.loc[excel_mask] = excel_dates

    # Bước 2 (Các định dạng ngày-đầu-tiên phổ biến):
    dayfirst_formats = ['%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y', '%d-%m-%y', '%d/%m/%y', '%Y-%m-%d']
    
    for fmt in dayfirst_formats:
        # Chỉ cố gắng phân tích các giá trị vẫn còn là NaT
        remaining_mask = parsed_dates.isna()
        if not remaining_mask.any():
            break # Tất cả đã được phân tích
        
        try:
            temp_parsed = pd.to_datetime(s[remaining_mask], format=fmt, errors='coerce')
            parsed_dates.loc[remaining_mask] = parsed_dates.loc[remaining_mask].fillna(temp_parsed)
        except Exception:
            # Continue to next format if there's an issue with the format
            pass

    # Bước 3 (Thử lại tổng quát): Đối với bất kỳ giá trị nào vẫn còn NaT, thử với dayfirst=True
    remaining_mask = parsed_dates.isna()
    if remaining_mask.any():
        general_parsed = pd.to_datetime(s[remaining_mask], dayfirst=True, errors='coerce')
        parsed_dates.loc[remaining_mask] = parsed_dates.loc[remaining_mask].fillna(general_parsed)

    # Làm sạch: Chuẩn hóa để loại bỏ thông tin thời gian
    return parsed_dates.dt.normalize()

def load_csv(path: Path) -> pd.DataFrame:
    """
    Tải tệp CSV đầu vào, làm sạch dữ liệu, chuẩn hóa tên cột, phát hiện và chuẩn hóa cột ngày,
    và đảm bảo sự tồn tại của các cột số No1 đến No27.
    """
    if not path.exists():
        raise FileNotFoundError(f"Tệp đầu vào không tìm thấy tại: {path}")

    # Đọc CSV với dtype=str để giữ tất cả các giá trị dưới dạng chuỗi và tránh chuyển đổi NA mặc định
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # Làm sạch tên cột: loại bỏ ký tự BOM và khoảng trắng thừa
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    # Xử lý dòng tiêu đề phụ nếu tồn tại cột "STT"
    if 'STT' in df.columns and not df.empty:
        # Check if the first data row (index 0 after header read) contains typical sub-header values
        sub_header_indicators = {"STT", "No.", "No"}
        if df.iloc[0]['STT'] in sub_header_indicators:
            df = df.iloc[1:].copy() # Drop the sub-header row
            df.reset_index(drop=True, inplace=True)

    # Phát hiện và chuẩn hóa cột ngày
    date_col_found = False
    if 'Date' in df.columns:
        df['Date'] = parse_date_dayfirst(df['Date'])
        date_col_found = True
    else:
        # Tìm cột có tỷ lệ giá trị ngày hợp lệ cao nhất
        best_date_col = None
        max_valid_dates = -1
        
        for col in df.columns:
            # Skip if column is entirely empty strings or just whitespace
            if df[col].astype(str).str.strip().eq('').all():
                continue
            
            parsed = parse_date_dayfirst(df[col])
            valid_dates_count = parsed.count() # Count non-NaT values
            
            if valid_dates_count > max_valid_dates:
                max_valid_dates = valid_dates_count
                best_date_col = col
        
        if best_date_col:
            df['Date'] = parse_date_dayfirst(df[best_date_col])
            date_col_found = True
        else:
            raise ValueError("Không thể tìm thấy cột ngày hợp lệ trong tệp CSV.")
    
    if not date_col_found:
        raise ValueError("Không thể tìm thấy hoặc phân tích cú pháp cột ngày hợp lệ.")

    # Đảm bảo cột số: Thêm cột NoX nếu thiếu
    for col in NO_COLS:
        if col not in df.columns:
            df[col] = '' # Thêm cột với giá trị rỗng

    # Loại bỏ các dòng có giá trị "Date" là NaT (không hợp lệ)
    df.dropna(subset=['Date'], inplace=True)
    
    if df.empty:
        raise ValueError("Không có dữ liệu hợp lệ sau khi phân tích ngày.")

    # Sắp xếp DataFrame theo "Date"
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Trả về một DataFrame mới chỉ chứa cột "Date" và các cột No1 đến No27
    return df[['Date'] + NO_COLS]

def count_digits_0_9(row: pd.Series) -> dict:
    """
    Đối với một dòng dữ liệu (tương ứng với một ngày), đếm tần suất xuất hiện của
    từng chữ số (0-9) ở hàng chục và hàng đơn vị của các số trong các cột No1 đến No27.
    """
    counts = {str(i): 0 for i in range(10)}

    for col_name in NO_COLS:
        value = str(row[col_name]).strip()
        if not value:
            continue

        processed = False
        try:
            # Cố gắng chuyển đổi thành số nguyên
            num = int(value)
            # Định dạng thành chuỗi hai chữ số (e.g., 5 -> "05", 12 -> "12")
            num_str = f'{num:02d}'
            if len(num_str) >= 2:
                tens_digit = num_str[-2]
                units_digit = num_str[-1]
                counts[tens_digit] += 1
                counts[units_digit] += 1
                processed = True
        except ValueError:
            # Nếu không thể chuyển đổi thành số nguyên, trích xuất chữ số
            digits = re.findall(r'\d', value)
            if len(digits) >= 2:
                tens_digit = digits[-2]
                units_digit = digits[-1]
                counts[tens_digit] += 1
                counts[units_digit] += 1
                processed = True
        
    return counts

def make_rows_for_date(date_val: pd.Timestamp, counts: dict) -> list:
    """
    Tạo 10 dòng dữ liệu đầu ra (Min1-Min5, Max1-Max5) cho một ngày cụ thể,
    dựa trên tần suất chữ số đã đếm được.
    """
    out_rows = []
    
    # Lấy các mức tần suất duy nhất và sắp xếp
    unique_freqs = sorted(list(set(counts.values())))
    
    asc_freqs = unique_freqs
    desc_freqs = sorted(unique_freqs, reverse=True)

    def create_output_row(label: str, freq: int):
        row_dict = {
            "Date": date_val,
            "D2": label,
            "Freq": freq,
            "Count": 0 
        }
        digit_count_for_freq = 0
        
        for digit in range(10):
            s_digit = str(digit)
            if counts.get(s_digit) == freq:
                row_dict[s_digit] = s_digit
                digit_count_for_freq += 1
            else:
                row_dict[s_digit] = '' # Để trống nếu chữ số không có tần suất đó
        
        row_dict["Count"] = digit_count_for_freq
        return row_dict

    # Tạo dòng Min1-Min5
    for i in range(5):
        if i < len(asc_freqs):
            freq = asc_freqs[i]
            out_rows.append(create_output_row(f"Min{i+1}", freq))
        else:
            # Điền các dòng rỗng nếu không đủ 5 mức tần suất
            out_rows.append({
                "Date": date_val, "D2": f"Min{i+1}", "Freq": '', "Count": '',
                **{str(d): '' for d in range(10)}
            })

    # Tạo dòng Max1-Max5
    for i in range(5):
        if i < len(desc_freqs):
            freq = desc_freqs[i]
            out_rows.append(create_output_row(f"Max{i+1}", freq))
        else:
            # Điền các dòng rỗng nếu không đủ 5 mức tần suất
            out_rows.append({
                "Date": date_val, "D2": f"Max{i+1}", "Freq": '', "Count": '',
                **{str(d): '' for d in range(10)}
            })
            
    return out_rows

@app.route('/process_csv', methods=['POST'])
def process_csv_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        # Use TemporaryDirectory to ensure cleanup of temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv_path = Path(tmpdir) / file.filename
            output_csv_path = Path(tmpdir) / 'D2.csv'
            
            # Save the uploaded file temporarily
            file.save(input_csv_path)
            
            try:
                # Load and preprocess the input CSV
                df_input = load_csv(input_csv_path)
                
                all_output_rows = []
                
                # Process each row (day) in the input DataFrame
                for index, row in df_input.iterrows():
                    date_val = row['Date']
                    digit_counts = count_digits_0_9(row)
                    date_output_rows = make_rows_for_date(date_val, digit_counts)
                    all_output_rows.extend(date_output_rows)
                
                # Create final DataFrame from all generated output rows
                output_cols = ["Date", "D2", "Freq", "Count"] + [str(i) for i in range(10)]
                df_output = pd.DataFrame(all_output_rows, columns=output_cols)
                
                # Format Date column for output CSV
                df_output['Date'] = df_output['Date'].dt.strftime("%d-%m-%Y")
                
                # Export to D2.csv
                df_output.to_csv(output_csv_path, index=False, encoding='utf-8')
                
                # Send the generated D2.csv back to the client
                return send_file(output_csv_path, mimetype='text/csv', as_attachment=True, download_name='D2.csv')
                
            except FileNotFoundError as e:
                return jsonify({"error": str(e)}), 404
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                app.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

@app.route('/')
def home():
    """
    Root endpoint for the API. Provides a simple status message.
    """
    return jsonify({"message": "CSV Digit Frequency Analyzer API is running. Upload CSV to /process_csv"}), 200

if __name__ == '__main__':
    # For local development, run with debug=True.
    # On Render.com, Gunicorn will manage the app, and PORT will be set by environment.
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
import pandas as pd
import numpy as np
import re
from pathlib import Path
import os
import tempfile
import shutil
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# --- Configuration ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Use temporary directories for uploads and outputs
UPLOAD_DIR = Path(tempfile.mkdtemp())
OUTPUT_DIR = Path(tempfile.mkdtemp())

# Ensure temporary directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# List of columns to process (No1 to No27)
NO_COLS = [f"No{i}" for i in range(1, 28)]

# --- Core Logic Functions ---

def parse_date_dayfirst(s: pd.Series) -> pd.Series:
    """
    Phân tích cú pháp một pd.Series thành các đối tượng datetime64[ns] một cách mạnh mẽ,
    ưu tiên định dạng ngày-đầu-tiên (day-first).
    """
    s = s.apply(lambda x: str(x).strip() if pd.notna(x) else None)
    
    # Bước 1 (Excel Serial Dates):
    # Cố gắng chuyển đổi các giá trị số (trong khoảng từ 1 đến 80000)
    # thành ngày tháng theo định dạng số serial của Excel (gốc là 1899-12-30).
    excel_serial_mask = s.str.match(r'^\d+$', na=False)
    numeric_s = pd.to_numeric(s[excel_serial_mask], errors='coerce')
    
    excel_dates = pd.NaT
    if not numeric_s.dropna().empty:
        # Excel's epoch is 1899-12-30. Pandas default is 1970-01-01.
        # Days from 1899-12-30 to 1970-01-01 is 25569.
        # Add a day for Windows Excel serial numbers (Excel for Windows starts at day 1 for 1900-01-01, Mac at 1904-01-01).
        # Adjust for 1900 leap year bug in Excel.
        # A simple approach: use pd.to_datetime with unit='D' and origin.
        origin_date = pd.Timestamp('1899-12-30')
        excel_dates = pd.to_datetime(numeric_s.dropna(), unit='D', origin=origin_date, errors='coerce')
        # Filter for reasonable date ranges if necessary, here we rely on errors='coerce'

    result = pd.Series(pd.NaT, index=s.index)
    result.loc[excel_serial_mask] = excel_dates

    # Bước 2 (Các định dạng ngày-đầu-tiên phổ biến):
    dayfirst_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%d-%m-%y", "%d/%m/%y", "%Y-%m-%d" # %Y-%m-%d is not strictly day-first, but common
    ]
    
    remaining_mask = result.isna()
    remaining_s = s[remaining_mask].replace({'nan': None, 'NaT': None, '': None, None: None})

    for fmt in dayfirst_formats:
        if remaining_s.dropna().empty:
            break
        
        parsed_current = pd.to_datetime(remaining_s, format=fmt, errors='coerce')
        result.loc[remaining_mask] = parsed_current
        remaining_mask = result.isna()
        remaining_s = s[remaining_mask].replace({'nan': None, 'NaT': None, '': None, None: None})

    # Bước 3 (Thử lại tổng quát):
    if not remaining_s.dropna().empty:
        result.loc[remaining_mask] = pd.to_datetime(remaining_s, dayfirst=True, errors='coerce')

    # Làm sạch: Chuẩn hóa ngày để loại bỏ thông tin thời gian
    result = result.dt.normalize()
    return result

def load_csv(path: Path) -> pd.DataFrame:
    """
    Tải tệp CSV đầu vào, làm sạch dữ liệu, chuẩn hóa tên cột, phát hiện và chuẩn hóa cột ngày,
    và đảm bảo sự tồn tại của các cột số No1 đến No27.
    """
    if not path.exists():
        raise FileNotFoundError(f"Tệp đầu vào không tìm thấy: {path}")

    # Đọc CSV với dtype=str để giữ tất cả các giá trị dưới dạng chuỗi ban đầu
    # và keep_default_na=False để tránh chuyển đổi các chuỗi rỗng thành NaN mặc định.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # Làm sạch tên cột: loại bỏ ký tự BOM (\ufeff) và khoảng trắng thừa.
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()

    # Xử lý dòng tiêu đề phụ: Nếu cột "STT" tồn tại và có các giá trị như "STT", "No.", "No"
    if "STT" in df.columns:
        sub_header_mask = df["STT"].isin(["STT", "No.", "No"])
        if sub_header_mask.any():
            df = df[~sub_header_mask].reset_index(drop=True)

    # Phát hiện và chuẩn hóa cột ngày
    date_col_name = None
    if "Date" in df.columns:
        df["Date"] = parse_date_dayfirst(df["Date"])
        date_col_name = "Date"
    else:
        # Tự động dò tìm cột có tỷ lệ giá trị ngày hợp lệ cao nhất
        best_date_col = None
        max_valid_dates = -1
        for col in df.columns:
            temp_parsed_dates = parse_date_dayfirst(df[col])
            valid_dates_count = temp_parsed_dates.notna().sum()
            
            # Chỉ xem xét các cột có ít nhất một ngày hợp lệ
            if valid_dates_count > max_valid_dates and valid_dates_count > 0:
                max_valid_dates = valid_dates_count
                best_date_col = col
        
        if best_date_col:
            df["Date"] = parse_date_dayfirst(df[best_date_col])
            date_col_name = "Date"
            if best_date_col != "Date":
                # Only drop if the detected column is not named "Date" already
                # and it's not one of the NO_COLS
                if best_date_col not in NO_COLS:
                    df = df.drop(columns=[best_date_col])
        else:
            raise ValueError("Không tìm thấy cột ngày hợp lệ trong tệp CSV.")

    if not date_col_name:
        raise ValueError("Cột 'Date' không tìm thấy và không thể suy luận.")

    # Đảm bảo cột số No1 đến No27
    for col in NO_COLS:
        if col not in df.columns:
            df[col] = "" # Thêm cột với giá trị rỗng

    # Loại bỏ các dòng có giá trị "Date" là NaT (không hợp lệ)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)

    # Sắp xếp DataFrame theo "Date" và đặt lại chỉ mục.
    df = df.sort_values(by="Date").reset_index(drop=True)

    # Chỉ trả về cột "Date" và các cột No1 đến No27
    return df[["Date"] + NO_COLS]

def count_digits_0_9(row: pd.Series) -> dict:
    """
    Đối với một dòng dữ liệu (tương ứng với một ngày), đếm tần suất xuất hiện của từng chữ số (0-9)
    trong các cột No1 đến No27.
    """
    counts = {str(i): 0 for i in range(10)}

    for col in NO_COLS:
        value = row.get(col)
        if pd.isna(value) or value is None or str(value).strip() == "":
            continue

        s_val = str(value).strip()

        # Cố gắng chuyển đổi thành số nguyên
        try:
            # Handle float strings like "12.0"
            num = int(float(s_val))
            s_num = f"{num:02d}" # Định dạng thành chuỗi hai chữ số (ví dụ: 5 -> "05")
            if len(s_num) >= 2:
                tens_digit = s_num[-2]
                units_digit = s_num[-1]
                if tens_digit.isdigit(): counts[tens_digit] += 1
                if units_digit.isdigit(): counts[units_digit] += 1
        except ValueError:
            # Không thể chuyển đổi thành số nguyên, trích xuất chữ số từ chuỗi
            digits_in_string = re.findall(r'\d', s_val)
            if len(digits_in_string) >= 2:
                # Lấy hai chữ số cuối cùng
                tens_digit = digits_in_string[-2]
                units_digit = digits_in_string[-1]
                counts[tens_digit] += 1
                counts[units_digit] += 1
            # Nếu ít hơn hai chữ số, bỏ qua (không làm gì)
    return counts

def make_rows_for_date(date_val: pd.Timestamp, counts: dict) -> list:
    """
    Tạo 10 dòng dữ liệu đầu ra (Min1-Min5, Max1-Max5) cho một ngày cụ thể,
    dựa trên tần suất chữ số đã đếm được.
    """
    output_rows = []

    # Xác định các mức tần suất duy nhất
    unique_frequencies = sorted(list(set(counts.values())))

    # Hàm trợ giúp để tạo một dòng đầu ra
    def make_row(label: str, freq: int):
        row_dict = {
            "Date": date_val,
            "D2": label,
            "Freq": freq,
            "Count": 0 # Số lượng chữ số có tần suất này
        }
        for digit in range(10):
            str_digit = str(digit)
            # Điền chữ số nếu tần suất của nó bằng freq, nếu không để trống
            row_dict[str_digit] = str_digit if counts.get(str_digit, 0) == freq else ""
            if counts.get(str_digit, 0) == freq:
                row_dict["Count"] += 1
        return row_dict

    # Tạo dòng Min1-Min5
    for i in range(5):
        if i < len(unique_frequencies):
            current_freq = unique_frequencies[i]
            output_rows.append(make_row(f"Min{i+1}", current_freq))
        else:
            # Điền các dòng trống nếu không đủ 5 mức tần suất duy nhất
            empty_row_dict = {
                "Date": date_val, "D2": f"Min{i+1}", "Freq": "", "Count": ""
            }
            for digit in range(10):
                empty_row_dict[str(digit)] = ""
            output_rows.append(empty_row_dict)

    # Tạo dòng Max1-Max5 (lấy 5 mức tần suất cao nhất)
    # Lật ngược unique_frequencies để có thứ tự giảm dần cho Max
    desc_frequencies = sorted(unique_frequencies, reverse=True)
    for i in range(5):
        if i < len(desc_frequencies):
            current_freq = desc_frequencies[i]
            output_rows.append(make_row(f"Max{i+1}", current_freq))
        else:
            # Điền các dòng trống nếu không đủ 5 mức tần suất duy nhất
            empty_row_dict = {
                "Date": date_val, "D2": f"Max{i+1}", "Freq": "", "Count": ""
            }
            for digit in range(10):
                empty_row_dict[str(digit)] = ""
            output_rows.append(empty_row_dict)

    return output_rows

# --- Flask Endpoints ---

@app.route('/process_csv', methods=['POST'])
def process_csv():
    if 'lucky_csv' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['lucky_csv']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        input_filepath = UPLOAD_DIR / "Lucky.csv"
        output_filepath = OUTPUT_DIR / "D2.csv"
        
        try:
            # Save uploaded file
            file.save(input_filepath)

            # Load and process data
            df_input = load_csv(input_filepath)

            out_rows = []
            for _, row in df_input.iterrows():
                date_val = row["Date"]
                counts = count_digits_0_9(row)
                out_rows.extend(make_rows_for_date(date_val, counts))

            # Create final DataFrame
            output_df = pd.DataFrame(out_rows)
            
            # Ensure all digit columns are present
            all_output_cols = ["Date", "D2", "Freq", "Count"] + [str(i) for i in range(10)]
            for col in [str(i) for i in range(10)]:
                if col not in output_df.columns:
                    output_df[col] = "" # Add missing digit columns
            
            output_df = output_df[all_output_cols]

            # Định dạng lại cột "Date"
            output_df["Date"] = output_df["Date"].dt.strftime("%d-%m-%Y")

            # Xuất DataFrame ra tệp D2.csv
            output_df.to_csv(output_filepath, index=False, encoding="utf-8")

            return jsonify({"message": "Xử lý thành công! Tệp D2.csv đã sẵn sàng để tải xuống."}), 200

        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Lỗi trong quá trình xử lý: {e}"}), 500

@app.route('/download_d2_csv', methods=['GET'])
def download_d2_csv():
    output_filepath = OUTPUT_DIR / "D2.csv"
    if not output_filepath.exists():
        return jsonify({"error": "Tệp D2.csv không tìm thấy. Vui lòng xử lý CSV trước."}), 404
    
    try:
        return send_file(str(output_filepath), as_attachment=True, download_name="D2.csv", mimetype='text/csv')
    except Exception as e:
        return jsonify({"error": f"Lỗi khi tải tệp: {e}"}), 500

@app.route('/')
def home():
    return "Backend API for CSV processing is running. Use /process_csv and /download_d2_csv endpoints."

# Cleanup temporary directories on app shutdown (for local testing mostly)
@app.teardown_appcontext
def cleanup_temp_dirs(exception=None):
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == '__main__':
    # For local development, create a dummy Lucky.csv if it doesn't exist for testing purposes
    # This block is for local testing only, not for deployment logic
    if not (Path(__file__).parent / "CSV").exists():
        (Path(__file__).parent / "CSV").mkdir()
    
    dummy_input_path = Path(__file__).parent / "CSV" / "Lucky.csv"
    if not dummy_input_path.exists():
        print(f"Creating a dummy CSV for testing at {dummy_input_path}")
        dummy_data = {
            "Date": ["01/01/2023", "02-01-2023", "03.01.2023", "2023-01-04", "05/Jan/2023"],
            "No1": ["1", "5", "10", "25", "3a"],
            "No2": ["2", "7", "11", "26", "b12"],
            "No3": ["3", "8", "12", "27", "1c3"],
            "No4": ["4", "9", "13", "28", "abc"]
        }
        pd.DataFrame(dummy_data).to_csv(dummy_input_path, index=False)

    app.run(debug=True, port=5000)
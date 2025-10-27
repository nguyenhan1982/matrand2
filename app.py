import pandas as pd
import numpy as np
from pathlib import Path
import os
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Cấu trúc và Phong cách Mã hóa đã được áp dụng trong tất cả các hàm

# Hằng số Toàn cục (sẽ là cục bộ hoặc truyền vào hàm trong Flask context)
NO_COLS = [f"No{i}" for i in range(1, 28)]

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

@app.route('/')
def hello_world():
    """Endpoint kiểm tra API."""
    return jsonify({"message": "CSV Processing API is running!"})

# Các Hàm Chính và Vai trò của chúng (như được mô tả trong yêu cầu)

def parse_date_dayfirst(s: pd.Series) -> pd.Series:
    """
    Mục đích: Một hàm mạnh mẽ để phân tích chuỗi ngày tháng từ một pd.Series thành datetime64[ns],
    ưu tiên định dạng ngày-tháng-năm và xử lý các định dạng không nhất quán.
    """
    # Bước 1: Xử lý Excel serial dates
    # Cố gắng chuyển đổi thành số, nếu thành công và trong phạm vi hợp lý, coi là Excel serial date.
    numeric_s = pd.to_numeric(s, errors='coerce')
    excel_serial_mask = numeric_s.notna() & (numeric_s > 1) & (numeric_s < 80000)
    
    # Khởi tạo Series kết quả với NaT
    parsed_dates = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    
    # Chuyển đổi Excel serial dates
    parsed_dates.loc[excel_serial_mask] = pd.to_datetime(
        numeric_s.loc[excel_serial_mask], unit="d", origin="1899-12-30", errors="coerce"
    )

    # Chuẩn hóa các chuỗi rỗng/NaN thành None trước khi phân tích
    s_cleaned = s.astype(str).str.strip().replace({"": None, "NaT": None, "nan": None, "None": None})
    
    # Bước 2: Xử lý các định dạng ngày-tháng-năm thông thường
    # Ưu tiên dayfirst (d/m/Y)
    dayfirst_formats = ["%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y"]
    
    for fmt in dayfirst_formats:
        # Chỉ cố gắng phân tích các giá trị chưa được phân tích (vẫn là NaT)
        mask_to_parse = parsed_dates.isna()
        if mask_to_parse.any():
            parsed_dates.loc[mask_to_parse] = pd.to_datetime(
                s_cleaned.loc[mask_to_parse], format=fmt, errors="coerce"
            )

    # Bước 3: Xử lý định dạng năm-tháng-ngày (Y-m-d) nếu còn NaT
    mask_to_parse = parsed_dates.isna()
    if mask_to_parse.any():
        parsed_dates.loc[mask_to_parse] = pd.to_datetime(
            s_cleaned.loc[mask_to_parse], format="%Y-%m-%d", errors="coerce"
        )
        
    # Bước 4: Thử phân tích ngày tháng cuối cùng với dayfirst=True cho bất kỳ giá trị nào còn lại là NaT
    mask_to_parse = parsed_dates.isna()
    if mask_to_parse.any():
        parsed_dates.loc[mask_to_parse] = pd.to_datetime(
            s_cleaned.loc[mask_to_parse], dayfirst=True, errors="coerce"
        )

    # Luôn trả về ngày tháng chuẩn hóa (.dt.normalize()) trừ khi tất cả đều là NaT
    return parsed_dates.dt.normalize() if not parsed_dates.isna().all() else parsed_dates

def load_csv(path: Path) -> pd.DataFrame:
    """
    Mục đích: Tải file CSV đầu vào, làm sạch tiêu đề cột, xử lý các dòng tiêu đề phụ,
    chuẩn hóa cột ngày "Date" và đảm bảo tất cả các cột NO_COLS đều tồn tại.
    """
    if not path.exists():
        raise SystemExit(f"Lỗi: File CSV không tìm thấy tại {path}")

    # Đọc CSV với dtype=str và keep_default_na=False (rất quan trọng để tránh chuyển đổi chuỗi số thành NaN).
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # Làm sạch tên cột: Loại bỏ ký tự BOM (\ufeff) và strip() khoảng trắng.
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    # Bỏ dòng tiêu đề phụ nếu có
    if "STT" in df.columns:
        # Tạo một bản sao của DataFrame sau khi lọc để tránh SettingWithCopyWarning
        df = df[~df["STT"].isin(["STT", "No.", "No"])].copy()

    # Chuẩn hoá ngày
    if "Date" in df.columns:
        df["Date"] = parse_date_dayfirst(df["Date"].copy()) # Pass a copy to avoid SettingWithCopyWarning
    else:
        # tự dò cột ngày “giống nhất”
        date_col = None
        max_notna_ratio = -1
        # Lấy bản sao của DataFrame để tránh sửa đổi trong quá trình lặp
        df_copy_for_date_detection = df.copy() 
        for col in df_copy_for_date_detection.columns:
            try:
                # Tạo một bản sao của Series trước khi truyền vào hàm để tránh SettingWithCopyWarning
                parsed_series = parse_date_dayfirst(df_copy_for_date_detection[col].copy())
                notna_ratio = parsed_series.notna().mean()
                if notna_ratio > max_notna_ratio and notna_ratio > 0.1:  # Ít nhất 10% giá trị hợp lệ
                    max_notna_ratio = notna_ratio
                    date_col = col
            except Exception: # Bắt bất kỳ lỗi nào trong quá trình phân tích
                continue
        
        if date_col:
            df["Date"] = parse_date_dayfirst(df[date_col].copy()) # Pass a copy to avoid SettingWithCopyWarning
            if date_col != "Date":
                print(f"Cảnh báo: Cột '{date_col}' được sử dụng làm cột ngày.")
        else:
            raise ValueError("Không tìm thấy cột ngày hợp lệ trong file CSV.")
            
    # Đảm bảo có No1..No27
    for col in NO_COLS:
        if col not in df.columns:
            df[col] = "" # Thêm cột với giá trị mặc định là chuỗi rỗng

    # Lọc bỏ các hàng có "Date" là NaT, sắp xếp theo "Date", và reset index.
    df = df.dropna(subset=["Date"]).sort_values(by="Date").reset_index(drop=True).copy()

    # Trả về một DataFrame mới chỉ chứa cột "Date" và NO_COLS đã được .copy().
    return df[["Date"] + NO_COLS].copy()


def count_digits_0_9(row: pd.Series) -> dict:
    """
    Mục đích: Đếm tần suất xuất hiện của mỗi chữ số (0-9) trong các giá trị của các cột NO_COLS
    trong một hàng DataFrame cụ thể.
    """
    counts = {str(d): 0 for d in range(10)}
    for col in NO_COLS:
        value = str(row[col]).strip()
        if not value:
            continue
        try:
            # Cố gắng chuyển đổi thành số nguyên
            iv = int(value)
            # Định dạng nó thành chuỗi 2 chữ số (ví dụ: 5 -> "05", 12 -> "12")
            s_iv = f"{iv:02d}"
            for digit_char in s_iv:
                if digit_char.isdigit():
                    counts[digit_char] += 1
        except ValueError:
            # Nếu chuyển đổi int() thất bại (ví dụ: chuỗi có ký tự không phải số)
            # Trích xuất tất cả các chữ số từ chuỗi
            digits = [char for char in value if char.isdigit()]
            if len(digits) >= 2:
                # Lấy 2 chữ số cuối cùng
                for digit_char in digits[-2:]:
                    counts[digit_char] += 1
    return counts

def make_rows_for_date(date_val: pd.Timestamp, counts: dict) -> list:
    """
    Mục đích: Tạo một danh sách các dictionary, mỗi dictionary đại diện cho một hàng dữ liệu đầu ra,
    dựa trên ngày và tần suất chữ số đã đếm. Các hàng này sẽ thể hiện tần suất "Min" và "Max" của các chữ số.
    """
    # Chuyển đổi counts thành pd.Series để dễ dàng lấy các tần suất độc nhất.
    freq_series = pd.Series(counts)
    
    # các mức tần suất phân biệt (chỉ những tần suất > 0)
    distinct_freqs = freq_series[freq_series > 0].value_counts().sort_index()
    asc_freqs = sorted(distinct_freqs.index)
    desc_freqs = sorted(distinct_freqs.index, reverse=True)

    def make_row(label: str, freq_val: int | str, is_empty: bool = False) -> dict:
        """
        Hàm trợ giúp lồng nhau tạo một dictionary hàng cho đầu ra.
        """
        row_dict = {
            "Date": date_val,
            "D2": label,
            "Freq": "" if is_empty else freq_val,
            "Count": "" if is_empty else int(distinct_freqs.get(freq_val, 0)), # Ensure Count is int
        }
        for d in range(10):
            if is_empty:
                row_dict[str(d)] = ""
            else:
                # Gán chữ số đó nếu nó có tần suất freq_val, nếu không thì gán chuỗi rỗng ""
                row_dict[str(d)] = str(d) if freq_series[str(d)] == freq_val else ""
        return row_dict

    output_rows = []

    # cho Min1..Min5
    for i in range(5):
        if i < len(asc_freqs):
            freq = asc_freqs[i]
            output_rows.append(make_row(f"Min{i+1}", freq))
        else:
            output_rows.append(make_row(f"Min{i+1}", "", is_empty=True))

    # cho Max1..Max5
    for i in range(5):
        if i < len(desc_freqs):
            freq = desc_freqs[i]
            output_rows.append(make_row(f"Max{i+1}", freq))
        else:
            output_rows.append(make_row(f"Max{i+1}", "", is_empty=True))

    return output_rows

def process_csv_data(input_csv_path: Path, output_csv_path: Path) -> int:
    """
    Hàm này gói gọn toàn bộ logic xử lý CSV từ yêu cầu.
    Tải dữ liệu, chuẩn hóa, phân tích tần suất, và lưu kết quả.
    """
    df = load_csv(input_csv_path)
    out_rows = []
    for _, row in df.iterrows():
        date_val = row["Date"]
        counts = count_digits_0_9(row)
        out_rows.extend(make_rows_for_date(date_val, counts))

    output_cols = ["Date", "D2", "Freq", "Count"] + [str(d) for d in range(10)]
    df_output = pd.DataFrame(out_rows, columns=output_cols)

    # Định dạng ngày cho đầu ra: Chuyển cột "Date" thành định dạng chuỗi `%d-%m-%Y`.
    df_output["Date"] = df_output["Date"].dt.strftime("%d-%m-%Y")

    df_output.to_csv(output_csv_path, index=False, encoding="utf-8")
    return len(df_output)

@app.route('/process-csv', methods=['POST'])
def process_uploaded_csv():
    """
    API endpoint để nhận file CSV, xử lý và trả về file CSV kết quả.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Không có phần file trong yêu cầu"}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"error": "Không có file được chọn"}), 400

    if uploaded_file:
        temp_input_path = None
        temp_output_path = None
        try:
            # Sử dụng tempfile để tạo file tạm thời an toàn
            # Input file
            temp_input_fd, temp_input_path = tempfile.mkstemp(suffix=".csv")
            os.close(temp_input_fd) # Đóng file descriptor ngay lập tức

            uploaded_file.save(temp_input_path)
            
            # Output file
            temp_output_fd, temp_output_path = tempfile.mkstemp(suffix=".csv")
            os.close(temp_output_fd) # Đóng file descriptor ngay lập tức

            # Gọi hàm xử lý chính
            lines_generated = process_csv_data(Path(temp_input_path), Path(temp_output_path))
            
            print(f"Đã tạo {lines_generated} dòng dữ liệu vào {temp_output_path}")

            # Trả về file đã xử lý
            return send_file(temp_output_path, 
                             mimetype='text/csv',
                             as_attachment=True,
                             download_name='D2_processed.csv')

        except SystemExit as e:
            # Bắt lỗi SystemExit từ load_csv (file không tồn tại, mặc dù đã kiểm tra exists)
            # Hoặc các lỗi thoát sớm khác.
            return jsonify({"error": str(e)}), 400
        except ValueError as e:
            # Bắt lỗi ValueError từ load_csv (không tìm thấy cột ngày)
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            # Bắt các lỗi khác trong quá trình xử lý
            print(f"Lỗi khi xử lý CSV: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Lỗi nội bộ khi xử lý file: {e}"}), 500
        finally:
            # Đảm bảo xóa các file tạm thời
            if temp_input_path and os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if temp_output_path and os.path.exists(temp_output_path):
                os.remove(temp_output_path)

if __name__ == "__main__":
    app.run(debug=True)
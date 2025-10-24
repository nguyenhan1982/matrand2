import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import io
import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import re
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Kích hoạt CORS cho tất cả các route để frontend từ Netlify có thể gọi được API

def parse_date_dayfirst(date_str):
    """
    Chuẩn hóa chuỗi ngày tháng sang định dạng datetime64[ns].
    Hỗ trợ định dạng số serial Excel và nhiều định dạng ngày phổ biến (dd/mm/yyyy, yyyy-mm-dd, v.v.).

    Args:
        date_str (str): Chuỗi ngày tháng cần chuẩn hóa.

    Returns:
        pd.Timestamp: Đối tượng ngày tháng đã chuẩn hóa hoặc pd.NaT nếu không thể parse.
    """
    if pd.isna(date_str) or str(date_str).strip() == '':
        return pd.NaT

    s_date = str(date_str).strip()

    # 1. Xử lý định dạng số serial Excel (giả định <= 80000 là số serial)
    try:
        excel_serial = float(s_date)
        if 0 < excel_serial <= 80000: # Số serial Excel hợp lệ thường dương và không quá lớn
            # Excel Windows sử dụng 1899-12-30 làm ngày gốc (ngày 1 là 1899-12-31)
            return pd.to_datetime(excel_serial, unit='D', origin='1899-12-30')
    except ValueError:
        pass # Không phải số, tiếp tục parse dạng chuỗi

    # 2. Thử các định dạng ngày phổ biến (day-first và year-first)
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",  # Day-first with 4-digit year
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",  # Year-first with 4-digit year
        "%d-%m-%y", "%d/%m/%y", "%d.%m.%y",  # Day-first with 2-digit year
        "%m-%d-%Y", "%m/%d/%Y",             # Month-first (đề bài không nói rõ, nhưng nên có)
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(s_date, format=fmt, errors='raise')
        except ValueError:
            continue
    
    # 3. Thử với infer_datetime_format nếu các định dạng cụ thể không thành công
    # infer_datetime_format có thể phát hiện nhiều định dạng nhưng kém hiệu quả hơn
    try:
        return pd.to_datetime(s_date, infer_datetime_format=True, dayfirst=True, errors='raise')
    except ValueError:
        return pd.NaT # Trả về NaT nếu không thể parse được

def process_lottery_data(input_csv_file):
    """
    Xử lý dữ liệu xổ số từ file CSV đã cung cấp và tạo ra file kết quả tổng hợp
    theo tần suất xuất hiện các chữ số 0-9.
    Đảm bảo luôn tạo ra 10 dòng dữ liệu mới cho mỗi ngày (5 Min, 5 Max),
    điền giá trị null/rỗng nếu không đủ tần suất duy nhất.

    Args:
        input_csv_file (file-like object): Đối tượng file CSV đầu vào (ví dụ: request.files['file']).

    Returns:
        tuple: (io.BytesIO, str) - Buffer chứa dữ liệu của file D2.csv và thông báo kết quả.
    Raises:
        ValueError: Nếu không tìm thấy cột ngày hợp lệ hoặc có lỗi trong quá trình xử lý.
    """
    logging.info("Bắt đầu xử lý dữ liệu xổ số...")

    # Đọc CSV ban đầu với header=None để dò tìm header và các cột
    df_raw = pd.read_csv(input_csv_file, header=None, dtype=str)

    # 1. Tìm dòng tiêu đề thực sự
    header_row_index = 0
    found_header = False
    for i, row in df_raw.iterrows():
        # Kiểm tra nếu bất kỳ cell nào trong dòng chứa 'No1', 'No.', 'STT', 'Date' để xác định header
        if any(pd.notna(cell) and (re.search(r'No\s*1', str(cell), re.IGNORECASE) or 
                                   'No.' in str(cell) or 
                                   str(cell).strip().lower() in ['stt', 'date', 'ngày']) 
               for cell in row):
            header_row_index = i
            found_header = True
            break
    
    # Đọc lại CSV với header đã tìm được. 
    # Dùng io.StringIO(df_raw.to_csv(index=False, header=False)) để tránh đọc file vật lý lần nữa.
    input_csv_file.seek(0) # Đặt con trỏ file về đầu để đọc lại
    df = pd.read_csv(input_csv_file, header=header_row_index, dtype=str)

    # Đảm bảo tên cột không có khoảng trắng thừa và chuẩn hóa
    df.columns = [str(col).strip() for col in df.columns]

    # Loại bỏ các dòng tiêu đề phụ nếu chúng xuất hiện trong dữ liệu (ví dụ: dòng 'STT', 'No.' lặp lại)
    df = df[~df.apply(lambda row: any(pd.notna(x) and str(x).strip().lower() in ['stt', 'no.', 'no'] for x in row), axis=1)]

    # 2. Dò tìm cột ngày (Date) có số lượng ngày hợp lệ nhiều nhất
    date_col = None
    best_date_col_name = None
    max_valid_dates = -1
    
    for col_name in df.columns:
        temp_dates = df[col_name].apply(parse_date_dayfirst)
        valid_date_count = temp_dates.notna().sum()
        
        # Chọn cột có nhiều ngày hợp lệ nhất, ưu tiên cột có tên gợi ý 'Date' hoặc 'Ngày'
        if valid_date_count > max_valid_dates or \
           (valid_date_count == max_valid_dates and best_date_col_name is not None and 
            str(col_name).lower() in ['date', 'ngày'] and str(best_date_col_name).lower() not in ['date', 'ngày']):
            max_valid_dates = valid_date_count
            best_date_col_name = col_name

    if not best_date_col_name or max_valid_dates == 0:
        raise ValueError("Không tìm thấy cột ngày hợp lệ nào trong file đầu vào.")

    df['Date'] = df[best_date_col_name].apply(parse_date_dayfirst)
    df = df.dropna(subset=['Date']) # Loại bỏ các dòng không có ngày hợp lệ
    logging.info(f"Đã xác định cột ngày là '{best_date_col_name}' với {max_valid_dates} ngày hợp lệ.")

    # 3. Đảm bảo tồn tại các cột No1 đến No27
    no_cols = [f'No{i}' for i in range(1, 28)]
    for col in no_cols:
        if col not in df.columns:
            df[col] = pd.NA # Thêm cột rỗng nếu thiếu
    
    # Chỉ giữ lại cột Date và các cột NoX đã chuẩn hóa tên
    # Sử dụng list comprehension để lọc và giữ đúng thứ tự các cột NoX
    df = df[['Date'] + [col for col in no_cols if col in df.columns] + [col for col in no_cols if col not in df.columns]]

    # 4. Sắp xếp theo ngày tăng dần
    df = df.sort_values(by='Date').reset_index(drop=True)
    logging.info(f"Đã tải và chuẩn hóa {len(df)} dòng dữ liệu.")

    # 5. Xử lý tần suất chữ số cho mỗi ngày
    results_list = []
    
    for idx, row in df.iterrows():
        current_date = row['Date']
        digit_counts = Counter() # Đếm tần suất các chữ số 0-9
        
        # Duyệt qua các cột No1-No27
        for col in no_cols:
            value = row[col]
            if pd.notna(value):
                s_value = str(value).strip()
                # Yêu cầu mới: Nếu số có 1 chữ số, thêm số 0 đằng trước (ví dụ 7 -> 07)
                # Chỉ chấp nhận các giá trị có thể coi là số nguyên dương (không có dấu âm)
                if re.match(r'^\d+$', s_value): 
                    try:
                        num_int = int(s_value)
                        # Giả định các số xổ số là từ 0 đến 99
                        if 0 <= num_int <= 99: 
                            padded_num_str = str(num_int).zfill(2) # Pad với '0' nếu là số có 1 chữ số
                            for digit_char in padded_num_str:
                                # digit_char sẽ luôn là '0'-'9'
                                digit_counts[int(digit_char)] += 1
                        # Nếu số nằm ngoài khoảng 0-99, nó sẽ bị bỏ qua như các giá trị không hợp lệ khác
                    except ValueError:
                        logging.warning(f"Giá trị '{s_value}' từ cột '{col}' không thể chuyển đổi thành số nguyên sau khi kiểm tra regex. Bỏ qua.")
                # Bỏ qua các ô trống hoặc giá trị không hợp lệ
        
        # Đảm bảo tất cả các chữ số từ 0-9 đều có mặt trong digit_counts, kể cả tần suất 0
        for i in range(10):
            digit_counts.setdefault(i, 0)

        # Nhóm các chữ số theo tần suất của chúng
        # freq_to_digits: {tần suất: [list_các_chữ_số_có_tần_suất_đó]}
        freq_to_digits = defaultdict(list)
        for digit, freq in digit_counts.items():
            freq_to_digits[freq].append(digit)
        
        # Sắp xếp các chữ số trong mỗi nhóm tần suất theo thứ tự tăng dần
        for freq in freq_to_digits:
            freq_to_digits[freq].sort()

        # Chuẩn bị danh sách tần suất và các chữ số tương ứng, đã sắp xếp
        # Cho Min: sắp xếp theo tần suất tăng dần
        sorted_min_data = sorted([(freq, digits) for freq, digits in freq_to_digits.items()], key=lambda item: item[0])
        # Cho Max: sắp xếp theo tần suất giảm dần
        sorted_max_data = sorted([(freq, digits) for freq, digits in freq_to_digits.items()], key=lambda item: item[0], reverse=True)


        # Tạo 5 dòng "Min" (tần suất ít nhất đến lớn hơn)
        for j in range(5): # Luôn tạo 5 dòng Min
            output_row = {
                'Date': current_date.strftime('%Y-%m-%d'), # Định dạng ngày tháng cho đầu ra
                'D2': f'Min{j+1}'
            }

            if j < len(sorted_min_data):
                # Sử dụng dữ liệu thực tế nếu có đủ tần suất duy nhất
                freq, digits_in_group = sorted_min_data[j]
                output_row['Freq'] = freq
                output_row['Count'] = len(digits_in_group)
                for d in range(10):
                    output_row[str(d)] = d if d in digits_in_group else ''
            else:
                # Điền giá trị rỗng nếu không đủ tần suất duy nhất
                output_row['Freq'] = ''
                output_row['Count'] = ''
                for d in range(10):
                    output_row[str(d)] = ''
            
            results_list.append(output_row)

        # Tạo 5 dòng "Max" (tần suất nhiều nhất đến nhỏ hơn)
        for j in range(5): # Luôn tạo 5 dòng Max
            output_row = {
                'Date': current_date.strftime('%Y-%m-%d'),
                'D2': f'Max{j+1}'
            }

            if j < len(sorted_max_data):
                # Sử dụng dữ liệu thực tế nếu có đủ tần suất duy nhất
                freq, digits_in_group = sorted_max_data[j]
                output_row['Freq'] = freq
                output_row['Count'] = len(digits_in_group)
                for d in range(10):
                    output_row[str(d)] = d if d in digits_in_group else ''
            else:
                # Điền giá trị rỗng nếu không đủ tần suất duy nhất
                output_row['Freq'] = ''
                output_row['Count'] = ''
                for d in range(10):
                    output_row[str(d)] = ''
            
            results_list.append(output_row)

    # 6. Ghi ra file CSV kết quả
    output_df = pd.DataFrame(results_list)
    
    # Đảm bảo thứ tự cột theo yêu cầu
    output_columns = ['Date', 'D2', 'Freq', 'Count'] + [str(d) for d in range(10)]
    output_df = output_df[output_columns]

    # Tạo buffer để trả về file mà không cần ghi vào đĩa
    output_buffer = io.BytesIO()
    output_df.to_csv(output_buffer, index=False, encoding='utf-8')
    output_buffer.seek(0) # Đặt con trỏ về đầu buffer để sẵn sàng đọc

    message = f"Đã tạo D2.csv với {len(output_df)} dòng."
    logging.info(message)
    return output_buffer, message

@app.route('/')
def index():
    """
    Endpoint mặc định, dùng để kiểm tra API đã chạy hay chưa.
    """
    return "Lottery Data Processor API đang chạy. Sử dụng endpoint /process_csv để tải lên file của bạn."

@app.route('/process_csv', methods=['POST'])
def handle_process_csv():
    """
    Endpoint để nhận file CSV đầu vào từ frontend, xử lý và trả về file CSV kết quả.
    """
    if 'file' not in request.files:
        logging.warning("Không có phần file trong request.")
        return jsonify({"error": "Không tìm thấy file trong yêu cầu"}), 400
    
    file = request.files['file']
    if file.filename == '':
        logging.warning("File không có tên.")
        return jsonify({"error": "Không có file nào được chọn"}), 400
    
    if file:
        try:
            # Truyền file-like object trực tiếp vào hàm xử lý
            output_buffer, message = process_lottery_data(file)
            
            # Trả về file D2.csv dưới dạng tệp đính kèm để tải xuống
            return send_file(
                output_buffer,
                mimetype='text/csv',
                as_attachment=True,
                download_name='D2.csv'
            )
        except ValueError as e:
            logging.error(f"Lỗi xử lý dữ liệu: {e}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logging.error(f"Lỗi nội bộ server khi xử lý CSV: {e}", exc_info=True)
            return jsonify({"error": f"Lỗi nội bộ server: {e}"}), 500

if __name__ == '__main__':
    # Khi triển khai trên Render.com, gunicorn sẽ gọi `app:app`, không cần chạy app.run() trực tiếp.
    # Dùng cho việc phát triển và thử nghiệm cục bộ.
    logging.info("Chạy Flask app ở chế độ cục bộ.")
    app.run(debug=True, host='0.0.0.0', port=5000)
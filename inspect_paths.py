import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def analyze_path_file(file_path, file_label):
    print(f"\n{'='*60}")
    print(f"📂 ĐANG PHÂN TÍCH: {file_label}")
    print(f"📍 Đường dẫn: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file tại {file_path}")
        return

    try:
        mat_data = sio.loadmat(file_path)
    except Exception as e:
        print(f"❌ Lỗi khi đọc file .mat: {e}")
        return

    # 1. Tìm biến chứa dữ liệu đường truyền
    # Các key thường gặp trong dataset này dựa trên code cũ
    possible_keys = ['Pz1', 'S', 'Primary_path', 'Secondary_path', 'primary_path', 'secondary_path']
    
    path_data = None
    key_found = ''

    # Ưu tiên tìm theo key quen thuộc
    for key in possible_keys:
        if key in mat_data:
            path_data = mat_data[key]
            key_found = key
            break
    
    # Nếu không thấy, lấy biến mảng đầu tiên không phải metadata
    if path_data is None:
        for key in mat_data:
            if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                path_data = mat_data[key]
                key_found = key
                break
    
    if path_data is None:
        print("⚠️ Không tìm thấy dữ liệu đường truyền hợp lệ trong file.")
        return

    # Chuyển về dạng mảng 1 chiều (Vector)
    path_data = np.squeeze(path_data)
    
    print(f"✅ Đã tìm thấy biến: '{key_found}'")
    print(f"📊 Kích thước (Số lượng mẫu): {path_data.shape[0]}")
    print(f"🔢 Thống kê: Max={np.max(path_data):.5f}, Min={np.min(path_data):.5f}, Mean={np.mean(path_data):.5f}")

    # 2. Xuất dữ liệu ra CSV
    csv_name = f"view_{file_label.lower().replace(' ', '_')}.csv"
    df = pd.DataFrame(path_data, columns=['Amplitude'])
    df.to_csv(csv_name, index_label='Sample_Index')
    print(f"💾 Đã xuất chi tiết ra file: '{csv_name}' (Mở bằng Excel để xem từng số)")

    # 3. Vẽ biểu đồ
    plt.figure(figsize=(10, 4))
    plt.plot(path_data, color='tab:blue', linewidth=1.5)
    plt.title(f'Impulse Response: {file_label} (Key: {key_found})')
    plt.xlabel('Samples (Time)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    # Cấu hình đường dẫn file (Sửa lại nếu file của bạn nằm chỗ khác)
    # Giả định file nằm trong thư mục 'data' như cấu trúc main.py trước đó
    pri_file = os.path.join('data', 'Primary_path.mat')
    sec_file = os.path.join('data', 'Secondary_path.mat')
    
    # Nếu file nằm ở thư mục gốc thì dùng dòng này:
    # pri_file = 'Primary_path.mat'
    # sec_file = 'Secondary_path.mat'

    analyze_path_file(pri_file, "Primary Path")
    analyze_path_file(sec_file, "Secondary Path")

if __name__ == "__main__":
    main()
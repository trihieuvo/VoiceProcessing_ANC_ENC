import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def explore_control_filters():
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    # Hãy sửa lại đường dẫn này nếu file của bạn nằm chỗ khác
    file_path = 'data/Pretrained_Control_filters.mat'
    
    print(f"\n{'='*60}")
    print(f"📂 ĐANG ĐỌC FILE: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file tại {file_path}")
        print("   -> Hãy kiểm tra lại xem file có nằm trong thư mục 'data' không?")
        return

    # 1. Load dữ liệu
    try:
        mat_data = sio.loadmat(file_path)
    except Exception as e:
        print(f"❌ Lỗi khi đọc file .mat: {e}")
        return
    
    # 2. Tìm biến chứa ma trận bộ lọc
    # Thường tên biến là 'Wc' hoặc 'Control_filter'
    filter_matrix = None
    key_name = ''
    
    # Ưu tiên tìm key 'Wc'
    if 'Wc' in mat_data:
        filter_matrix = mat_data['Wc']
        key_name = 'Wc'
    # Nếu không có, tìm biến mảng 2 chiều bất kỳ
    else:
        for key in mat_data:
            if not key.startswith('__'):
                val = mat_data[key]
                if isinstance(val, np.ndarray) and val.ndim == 2:
                    filter_matrix = val
                    key_name = key
                    break
    
    if filter_matrix is None:
        print("❌ Không tìm thấy ma trận bộ lọc hợp lệ (dạng 2 chiều) trong file.")
        return

    # Đảm bảo dữ liệu là mảng Numpy
    filter_matrix = np.array(filter_matrix)

    print(f"✅ Đã tìm thấy biến: '{key_name}'")
    print(f"📊 Kích thước ma trận: {filter_matrix.shape}")
    print(f"   -> Số lượng bộ lọc (Classes): {filter_matrix.shape[0]}")
    print(f"   -> Độ dài mỗi bộ lọc (Tabs):  {filter_matrix.shape[1]}")

    # 3. Xuất ra file Excel (CSV) để xem chi tiết
    csv_filename = 'view_filters.csv'
    
    # Tạo DataFrame
    df = pd.DataFrame(filter_matrix)
    # Đặt tên dòng và cột cho dễ nhìn
    df.index = [f'Filter_{i}' for i in range(filter_matrix.shape[0])]
    df.columns = [f'Tap_{i}' for i in range(filter_matrix.shape[1])]
    
    df.to_csv(csv_filename)
    print(f"\n💾 Đã xuất dữ liệu ra file: '{csv_filename}'")
    print("   -> Bạn hãy mở file này bằng Excel để xem từng con số cụ thể.")

    # 4. Vẽ biểu đồ trực quan hóa
    print("\n📈 Đang vẽ biểu đồ 15 bộ lọc đầu tiên...")
    
    num_filters = filter_matrix.shape[0]
    display_count = min(num_filters, 15) # Chỉ vẽ tối đa 15 cái để không rối
    
    # Tạo lưới biểu đồ (5 dòng x 3 cột)
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Tìm min/max chung để trục Y đồng bộ (dễ so sánh)
    y_min, y_max = np.min(filter_matrix), np.max(filter_matrix)
    
    for i in range(display_count):
        ax = axes[i]
        filter_coef = filter_matrix[i, :]
        
        ax.plot(filter_coef, color='tab:blue', linewidth=1)
        ax.set_title(f'Filter Index: {i} (Label {i})', fontsize=10)
        ax.set_ylim([y_min, y_max]) # Cố định trục Y
        ax.grid(True, alpha=0.5)
        
    # Ẩn các ô biểu đồ thừa nếu có
    for i in range(display_count, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle(f'Visualization of Control Filters ({key_name})', y=1.02, fontsize=16)
    plt.show()

if __name__ == "__main__":
    explore_control_filters()
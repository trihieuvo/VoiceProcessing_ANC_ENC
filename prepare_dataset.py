import os
import shutil
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import soundfile as sf

# --- CẤU HÌNH HỆ THỐNG ---
# Tự động chọn số nhân CPU (Chừa lại 1-2 nhân để máy không bị đơ)
NUM_WORKERS = max(1, os.cpu_count() - 2) 

# --- CÁC HÀM XỬ LÝ ---

def load_mat_data(path):
    """Load dữ liệu .mat và trả về numpy array"""
    try:
        data = sio.loadmat(path)
        # Tìm key chứa dữ liệu (không lấy key hệ thống __)
        for key in data:
            if not key.startswith('__') and isinstance(data[key], np.ndarray):
                # squeeze để loại bỏ các chiều dư thừa (ví dụ (N, 1) -> (N,))
                return np.squeeze(data[key])
    except Exception as e:
        print(f"❌ Lỗi load {path}: {e}")
        return None

def full_simulation(waveform, pri_path, sec_path, filters_matrix):
    """
    Mô phỏng khử ồn trên TOÀN BỘ file âm thanh để tìm bộ lọc tốt nhất.
    """
    # 1. Tạo tín hiệu giả lập (Vectorized - Rất nhanh)
    # Dis: Nhiễu tại điểm khử (Tín hiệu gốc đi qua đường truyền sơ cấp)
    Dis = signal.lfilter(pri_path, 1, waveform)
    
    # Fx: Tín hiệu tham chiếu (Tín hiệu gốc đi qua đường truyền thứ cấp)
    Fx = signal.lfilter(sec_path, 1, waveform)
    
    # KHÔNG CẮT NGẮN DỮ LIỆU NỮA
    # Sử dụng toàn bộ chiều dài của Dis và Fx
    
    min_mse = float('inf')
    best_idx = -1
    
    # 2. Duyệt qua từng bộ lọc có sẵn (15 cái)
    # filters_matrix có dạng (15, 1024)
    for i in range(filters_matrix.shape[0]):
        w = filters_matrix[i]
        
        # Tạo tín hiệu chống ồn y = w * Fx (Tích chập)
        # signal.lfilter tương đương với việc chạy bộ lọc FIR trên toàn bộ chuỗi
        y = signal.lfilter(w, 1, Fx)
        
        # Tính tín hiệu lỗi e = Dis - y
        e = Dis - y
        
        # Tính năng lượng lỗi trung bình (Mean Squared Error) trên toàn bộ file
        mse = np.mean(e**2)
        
        # Tìm bộ lọc có lỗi nhỏ nhất
        if mse < min_mse:
            min_mse = mse
            best_idx = i
            
    return best_idx

def process_single_file(args):
    """Hàm xử lý 1 file (được gọi bởi các nhân CPU con)"""
    file_path, output_root, pri_path, sec_path, filters_matrix = args
    
    try:
        # Đọc file âm thanh
        # sf.read trả về (data, samplerate)
        waveform, sr = sf.read(file_path)
        
        # Lưu ý: Nếu dataset của bạn chưa phải 16kHz, kết quả có thể bị sai lệch
        # Nếu cần thiết, bạn có thể thêm code resample tại đây.
        
        # Tìm nhãn tốt nhất (Dựa trên toàn bộ nội dung file)
        label = full_simulation(waveform, pri_path, sec_path, filters_matrix)
        
        # Copy file vào thư mục tương ứng
        file_name = os.path.basename(file_path)
        target_dir = os.path.join(output_root, str(label))
        
        # Tạo thư mục nếu chưa có (ví dụ: train/5)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy file
        shutil.copy2(file_path, os.path.join(target_dir, file_name))
        
        return 1 # Đánh dấu thành công
    except Exception as e:
        # print(f"Lỗi xử lý file {file_path}: {e}")
        return 0 # Đánh dấu thất bại

def main():
    # --- CẤU HÌNH ĐƯỜNG DẪN (HÃY SỬA LẠI CHO ĐÚNG MÁY BẠN) ---
    
    # Thư mục chứa dữ liệu gốc (lộn xộn)
    RAW_DATA_ROOT = r'D:\UTE\VoiceProcessing\project\DATA\Synthesized_Dataset'
    
    # Thư mục đích (nơi chứa dữ liệu đã phân loại)
    PROCESSED_ROOT = r'D:\UTE\VoiceProcessing\project\DATA\Processed_Dataset_For_Train_1'
    
    # Các file dữ liệu hệ thống ANC
    PRI_PATH_FILE = 'data/Primary_path.mat'
    SEC_PATH_FILE = 'data/Secondary_path.mat'
    FILTERS_FILE = 'data/Pretrained_Control_filters.mat'

    print(f"🚀 Bắt đầu chương trình phân loại dataset (Chế độ: Full Content Check)")
    print(f"cpu_count: {os.cpu_count()} | workers: {NUM_WORKERS}")
    
    # 1. Load dữ liệu môi trường
    print("\n[1/3] Loading MATLAB data files...")
    pri_path = load_mat_data(PRI_PATH_FILE)
    sec_path = load_mat_data(SEC_PATH_FILE)
    filters_matrix = load_mat_data(FILTERS_FILE)
    
    if pri_path is None or filters_matrix is None or sec_path is None:
        print("❌ LỖI: Không tìm thấy hoặc không đọc được file .mat trong thư mục 'data/'")
        return

    # 2. Chuẩn bị danh sách file cần xử lý
    subfolders = ['Training_data', 'Validate_data', 'Testing_data']
    target_names = ['train', 'val', 'test']

    for sub, target in zip(subfolders, target_names):
        input_dir = os.path.join(RAW_DATA_ROOT, sub)
        output_dir = os.path.join(PROCESSED_ROOT, target)
        
        if not os.path.exists(input_dir):
            print(f"⚠️ Bỏ qua: Không tìm thấy thư mục {input_dir}")
            continue
            
        # Lấy danh sách tất cả file .wav
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.wav')]
        
        if len(files) == 0:
            print(f"⚠️ Thư mục {sub} trống, bỏ qua.")
            continue

        print(f"\n[2/3] Đang xử lý tập: {sub} -> {target}")
        print(f"      Số lượng file: {len(files)}")
        print(f"      Đang chạy song song trên {NUM_WORKERS} nhân CPU...")
        
        # 3. Chạy đa luồng (Multiprocessing)
        # Tạo danh sách tham số để truyền vào hàm process_single_file
        tasks = []
        for f in files:
            tasks.append((f, output_dir, pri_path, sec_path, filters_matrix))
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # map hàm xử lý với danh sách tasks
            # dùng tqdm để hiện thanh tiến trình
            results = list(tqdm(executor.map(process_single_file, tasks), total=len(tasks), unit="file"))
            
        success_count = sum(results)
        print(f"✅ Hoàn thành tập {sub}: {success_count}/{len(files)} files thành công.")

    print("\n🎉 TẤT CẢ ĐÃ HOÀN TẤT!")
    print(f"📂 Dữ liệu đã được phân loại tại: {PROCESSED_ROOT}")

if __name__ == "__main__":
    # Bắt buộc phải có dòng này trên Windows để chạy đa luồng
    main()
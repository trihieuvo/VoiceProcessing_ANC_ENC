# --- 1. LỰA CHỌN MÔ HÌNH (BỘ LỌC) ---
NOISE_CLASS = 'Household_Appliance'
# NOISE_CLASS = 'TVnRadio'
# NOISE_CLASS = 'Vechicles'
# NOISE_CLASS = 'Verbal_Human'

# --- 2. THAM SỐ THUẬT TOÁN XỬ LÝ TIẾNG NÓI (DSP) ---
# Các tham số này bắt buộc phải có để các hàm trong data_tools.py hoạt động
SAMPLE_RATE = 16000      # Tần số lấy mẫu tín hiệu
N_FFT = 256              # Độ dài khung thực hiện biến đổi Fourier (STFT)
HOP_LENGTH_FFT = 128     # Khoảng cách bước nhảy giữa các khung (Overlap 50%)
SLICE_LENGTH = 16384     # Độ dài đoạn âm thanh AI xử lý mỗi lần (~1 giây)
TARGET_dBFS = - 30.0      # Mức âm lượng chuẩn hóa trước khi lọc

# --- 3. ĐƯỜNG DẪN HỆ THỐNG ---
MODEL = "FC"             # Kiến trúc mạng Fully Connected
MODEL_NAME = f'DDAE_{MODEL}_{NOISE_CLASS}'
# Trỏ trực tiếp vào thư mục backend mới của bạn
PATH_WEIGHTS = f'backend/model_files/{MODEL_NAME}.h5'

# --- 4. ĐẦU RA KẾT QUẢ ---
PATH_DIR_PREDICT_ROOT = './Predictions'
PATH_PREDICT_OUTPUT_NAME = 'denoise.wav'
# File: main.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

# --- IMPORT MODULES ---
from src.network import CNN, CNNRes
from src.fxnlms import FxNLMS, train_fxnlms_algorithm
from src.utils import load_audio_file, load_anc_paths, load_pretrained_filters, save_audio_file
from src.simulation import Disturbance_generation_from_real_noise

# --- CẤU HÌNH PHÁT ÂM THANH (Windows Only) ---
try:
    import winsound 
    def play_sound(filepath):
        # Stop âm thanh cũ trước khi phát mới
        winsound.PlaySound(None, winsound.SND_PURGE)
        winsound.PlaySound(filepath, winsound.SND_FILENAME | winsound.SND_ASYNC)
except ImportError:
    def play_sound(filepath):
        print(f"Hệ thống không hỗ trợ phát trực tiếp. Vui lòng mở file: {filepath}")

def main():
    print("=== ANC BENCHMARK: STANDARD vs CNN vs RESNET ===")

    # --- 1. CẤU HÌNH & TẠO THƯ MỤC ---
    DATA_DIR = 'data'
    AUDIO_DIR = 'input_audio'
    RESULT_DIR = 'result'
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Đường dẫn file dữ liệu
    pri_mat = os.path.join(DATA_DIR, 'Primary_path.mat')
    sec_mat = os.path.join(DATA_DIR, 'Secondary_path.mat')
    filter_mat = os.path.join(DATA_DIR, 'Pretrained_Control_filters.mat')
    
    # Đường dẫn 2 Model
    model_cnn_path = os.path.join(DATA_DIR, 'best_model.pth') # Model của bạn
    model_resnet_path = os.path.join(DATA_DIR, 'model.pth')   # Model gốc
    
    # File âm thanh đầu vào
    #noise_wav = os.path.join(AUDIO_DIR, 'Mix_Aircraft_Traffic.wav')
    # noise_wav = os.path.join(AUDIO_DIR, 'Aircraft.wav') 
    noise_wav = os.path.join(AUDIO_DIR, 'trial.wav') 

    if not os.path.exists(noise_wav):
        print(f"❌ Lỗi: Không tìm thấy file âm thanh tại {noise_wav}")
        return

    # --- 2. TẢI DỮ LIỆU HỆ THỐNG ---
    print("--- [1/6] Đang tải dữ liệu hệ thống ---")
    pri_path, sec_path = load_anc_paths(pri_mat, sec_mat)
    waveform, fs = load_audio_file(noise_wav)
    pretrained_W = load_pretrained_filters(filter_mat)
    
    # --- 3. TẠO MÔI TRƯỜNG GIẢ LẬP ---
    print("--- [2/6] Đang tạo tín hiệu mô phỏng (Original Noise) ---")
    # Dis: Nhiễu tại điểm khử (Chưa xử lý)
    # Fx: Tín hiệu tham chiếu
    Dis, Fx, Raw_Noise = Disturbance_generation_from_real_noise(
        fs=fs, Repet=0, wave_form=waveform, Pri_path=pri_path, Sec_path=sec_path
    )
    
    # 1. Save Original
    file_original = os.path.join(RESULT_DIR, '1_original.wav')
    save_audio_file(file_original, Dis, fs)

    # --- 4. CHẠY AI ĐỂ CHỌN FILTER (CNN & RESNET) ---
    print("--- [3/6] Đang chạy Deep Learning Inference ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    -> Device: {device}")

    # Chuẩn bị Input cho AI (Chuyển Numpy -> Tensor, cắt 1s đầu)
    if isinstance(Raw_Noise, np.ndarray):
        Raw_Noise_Tensor = torch.from_numpy(Raw_Noise).float()
    else:
        Raw_Noise_Tensor = Raw_Noise.float()
        
    input_slice = Raw_Noise_Tensor[:16000] if Raw_Noise_Tensor.shape[0] > 16000 else Raw_Noise_Tensor
    ai_input = input_slice.view(1, 1, -1).to(device) # Shape (1, 1, 16000)

    # A. Model CNN (Của bạn)
    idx_cnn = 0
    if os.path.exists(model_cnn_path):
        try:
            cnn_net = CNN(channels=[[16], [32], [64], [128]], 
                          conv_kernels=[7, 5, 3, 3], conv_strides=[1, 1, 1, 1], 
                          conv_padding=[3, 2, 1, 1], pool_padding=[0, 0, 0, 0], num_classes=15).to(device)
            cnn_net.load_state_dict(torch.load(model_cnn_path, map_location=device))
            cnn_net.eval()
            with torch.no_grad():
                idx_cnn = torch.argmax(cnn_net(ai_input)).item()
            print(f"    -> [My CNN] chọn Filter số: {idx_cnn}")
        except Exception as e:
            print(f"    -> [My CNN] Lỗi: {e}. Dùng mặc định 0.")
    else:
        print("    -> [My CNN] Không tìm thấy file best_model.pth")

    # B. Model ResNet (Gốc)
    idx_res = 0
    if os.path.exists(model_resnet_path):
        try:
            res_net =  CNNRes(channels=[[128], [128]*2], conv_kernels=[80, 3], 
                        conv_strides=[4, 1], conv_padding=[38, 1], pool_padding=[0, 0], num_classes=15).to(device)
            res_net.load_state_dict(torch.load(model_resnet_path, map_location=device))
            res_net.eval()
            with torch.no_grad():
                idx_res = torch.argmax(res_net(ai_input)).item()
            print(f"    -> [ResNet] chọn Filter số: {idx_res}")
        except Exception as e:
            print(f"    -> [ResNet] Lỗi: {e}. Dùng mặc định 0.")
    else:
        print("    -> [ResNet] Không tìm thấy file model.pth")

    # --- 5. CHẠY THUẬT TOÁN ANC (3 TRƯỜNG HỢP) ---
    print("--- [4/6] Đang khử nhiễu (Simulation) ---")
    step_size = 0.005 # Chỉnh nhỏ hơn (vd 0.0001) nếu đồ thị bị vỡ

    # Case 2: Standard FxNLMS (Start from 0)
    print("    -> Running Standard FxNLMS...")
    ctrl_std = FxNLMS(Len=1024)
    ctrl_std.Wc.data.zero_()
    err_std = train_fxnlms_algorithm(ctrl_std, Fx, Dis, Stepsize=step_size)
    file_std = os.path.join(RESULT_DIR, '2_standard_nlms.wav')
    save_audio_file(file_std, np.array(err_std), fs)

    # Case 3: Hybrid CNN (Start from idx_cnn)
    print(f"    -> Running Hybrid with My CNN (Filter {idx_cnn})...")
    ctrl_cnn = FxNLMS(Len=1024)
    ctrl_cnn.Wc.data = torch.from_numpy(pretrained_W[idx_cnn]).float().unsqueeze(0)
    err_cnn = train_fxnlms_algorithm(ctrl_cnn, Fx, Dis, Stepsize=step_size)
    file_cnn = os.path.join(RESULT_DIR, '3_hybrid_cnn.wav')
    save_audio_file(file_cnn, np.array(err_cnn), fs)

    # Case 4: Hybrid ResNet (Start from idx_res)
    print(f"    -> Running Hybrid with ResNet (Filter {idx_res})...")
    ctrl_res = FxNLMS(Len=1024)
    ctrl_res.Wc.data = torch.from_numpy(pretrained_W[idx_res]).float().unsqueeze(0)
    err_res = train_fxnlms_algorithm(ctrl_res, Fx, Dis, Stepsize=step_size)
    file_res = os.path.join(RESULT_DIR, '4_hybrid_resnet.wav')
    save_audio_file(file_res, np.array(err_res), fs)

    # --- 6. VẼ ĐỒ THỊ SO SÁNH (4 DÒNG) ---
    print("--- [5/6] Vẽ biểu đồ ---")
    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    plt.subplots_adjust(bottom=0.10, hspace=0.4, top=0.95)

    # 1. Original
    axs[0].plot(Dis.numpy(), color='black', alpha=0.6, linewidth=0.8)
    axs[0].set_title('1. Original Noise')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True, alpha=0.5)

    # 2. Standard
    axs[1].plot(err_std, color='tab:orange', linewidth=1)
    axs[1].set_title('2. Standard FxNLMS')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True, alpha=0.5)

    # 3. Hybrid CNN
    axs[2].plot(err_cnn, color='tab:green', linewidth=1)
    axs[2].set_title(f'3. Hybrid with CNN (Filter {idx_cnn})')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True, alpha=0.5)

    # 4. Hybrid ResNet
    axs[3].plot(err_res, color='tab:blue', linewidth=1)
    axs[3].set_title(f'4. Hybrid with RESNET (Filter {idx_res})')
    axs[3].set_ylabel('Amplitude')
    axs[3].set_xlabel('Samples')
    axs[3].grid(True, alpha=0.5)

    # --- 7. TẠO NÚT BẤM ---
    # Vị trí nút: [left, bottom, width, height]
    btn_ax1 = plt.axes([0.15, 0.02, 0.15, 0.04])
    btn1 = Button(btn_ax1, 'Play Original')
    btn1.on_clicked(lambda x: play_sound(file_original))

    btn_ax2 = plt.axes([0.35, 0.02, 0.15, 0.04])
    btn2 = Button(btn_ax2, 'Play Standard')
    btn2.on_clicked(lambda x: play_sound(file_std))

    btn_ax3 = plt.axes([0.55, 0.02, 0.15, 0.04])
    btn3 = Button(btn_ax3, 'Play My CNN')
    btn3.on_clicked(lambda x: play_sound(file_cnn))

    btn_ax4 = plt.axes([0.75, 0.02, 0.15, 0.04])
    btn4 = Button(btn_ax4, 'Play ResNet')
    btn4.on_clicked(lambda x: play_sound(file_res))

    print("✅ Hoàn tất! Đang hiển thị biểu đồ...")
    plt.show()

if __name__ == "__main__":
    main()
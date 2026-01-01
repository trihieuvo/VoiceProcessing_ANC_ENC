import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
import time
from datetime import datetime
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from tensorflow.keras.models import load_model

# --- IMPORT MODULES CỦA BẠN ---
from src.anc.network import CNN, CNNRes
from src.anc.utils import load_audio_file, load_anc_paths, load_pretrained_filters, save_audio_file
from src.anc.simulation import Disturbance_generation_from_real_noise
from src.enc import data_tools
from src.enc import config_params

# Cấu hình giao diện
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# --- HÀM TỐI ƯU HÓA THUẬT TOÁN ---
def train_fxnlms_numpy(initial_weights_tensor, ref_tensor, dist_tensor, step_size=0.001, len_filter=1024):
    if isinstance(initial_weights_tensor, torch.Tensor):
        w = initial_weights_tensor.detach().cpu().numpy().flatten()
    else:
        w = initial_weights_tensor.flatten()
    if len(w) == 0: w = np.zeros(len_filter)

    ref = ref_tensor.detach().cpu().numpy().flatten()
    dist = dist_tensor.detach().cpu().numpy().flatten()
    n_samples = len(dist)
    e_signal = np.zeros(n_samples)
    x_buffer = np.zeros(len_filter)
    
    for i in range(n_samples):
        x_buffer = np.roll(x_buffer, 1)
        x_buffer[0] = ref[i]
        y = np.dot(w, x_buffer)
        e = dist[i] - y
        e_signal[i] = e
        power = np.dot(x_buffer, x_buffer) + 1e-6
        normalized_step = step_size / power
        w += normalized_step * e * x_buffer
        
    return e_signal

def downsample_signal(signal, max_points=100000):
    """Giảm số lượng điểm vẽ để tránh lỗi Matplotlib khi file quá dài"""
    if len(signal) > max_points:
        step = len(signal) // max_points
        return signal[::step]
    return signal

# --- CLASS XỬ LÝ ÂM THANH (THAY THẾ PYGAME) ---
class AudioPlayer:
    def __init__(self):
        self.data = None
        self.fs = 16000
        self.current_idx = 0
        self.is_playing = False
        self.stream = None
        self.lock = threading.Lock()
        
    def load(self, filepath):
        self.stop()
        try:
            data, fs = sf.read(filepath)
            if len(data.shape) > 1: data = data[:, 0] # Mono
            self.data = data.astype(np.float32)
            self.fs = fs
            self.current_idx = 0
            return True
        except Exception as e:
            print(f"Lỗi load file: {e}")
            return False

    def play(self):
        if self.data is None: return
        self.stop() # Đảm bảo stream cũ dừng
        self.is_playing = True
        
        # Tính đoạn dữ liệu còn lại từ vị trí hiện tại
        data_to_play = self.data[self.current_idx:]
        
        # Dùng sd.play là cách đơn giản nhất thay vì OutputStream phức tạp
        # Nhược điểm là khó track vị trí chính xác realtime, nhưng đủ dùng cho app này
        sd.play(data_to_play, self.fs)
        
        # Bắt đầu thread đếm thời gian để cập nhật slider (nếu cần logic phức tạp hơn)
        self._start_time = time.time()
        self._start_idx = self.current_idx

    def pause(self):
        if self.is_playing:
            sd.stop()
            self.is_playing = False
            # Cập nhật vị trí hiện tại dựa trên thời gian đã trôi qua
            elapsed = time.time() - self._start_time
            self.current_idx = self._start_idx + int(elapsed * self.fs)
            if self.current_idx > len(self.data):
                self.current_idx = 0

    def stop(self):
        sd.stop()
        self.is_playing = False
        # Không reset current_idx về 0 ở đây để hỗ trợ pause, 
        # logic reset sẽ nằm ở seek hoặc load

    def seek(self, ratio):
        if self.data is None: return
        self.current_idx = int(len(self.data) * ratio)
        if self.is_playing:
            self.play() # Restart từ vị trí mới

    def get_duration(self):
        if self.data is None: return 0
        return len(self.data) / self.fs

    def get_current_time(self):
        if self.data is None: return 0
        if self.is_playing:
            elapsed = time.time() - self._start_time
            curr = self._start_idx + int(elapsed * self.fs)
            return min(curr, len(self.data)) / self.fs
        else:
            return self.current_idx / self.fs

# --- LOGIC GHI ÂM ---
class AudioRecorder:
    def __init__(self, output_folder):
        self.recording = False
        self.output_folder = output_folder
        self.fs = 16000
        self.data = []
        os.makedirs(self.output_folder, exist_ok=True)

    def start(self):
        self.recording = True
        self.data = []
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self):
        with sd.InputStream(samplerate=self.fs, channels=1, callback=self._callback):
            while self.recording:
                sd.sleep(100)

    def _callback(self, indata, frames, time, status):
        if self.recording:
            self.data.append(indata.copy())

    def stop(self):
        self.recording = False
        if not self.data: return None
        recording_array = np.concatenate(self.data, axis=0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.wav"
        filepath = os.path.join(self.output_folder, filename)
        sf.write(filepath, recording_array, self.fs)
        return filepath

# --- LOGIC ANC ---
class ANCLogic:
    def run_anc(self, noise_wav_path):
        pri_mat = os.path.join('data', 'anc', 'Primary_path.mat')
        sec_mat = os.path.join('data', 'anc', 'Secondary_path.mat')
        filter_mat = os.path.join('data', 'anc', 'Pretrained_Control_filters.mat')
        model_cnn_path = os.path.join('data', 'anc', 'best_model.pth')
        model_resnet_path = os.path.join('data', 'anc', 'model.pth')
        result_dir = os.path.join('result', 'anc')
        os.makedirs(result_dir, exist_ok=True)

        pri_path, sec_path = load_anc_paths(pri_mat, sec_mat)
        waveform, fs = load_audio_file(noise_wav_path)
        pretrained_W = load_pretrained_filters(filter_mat)

        Dis, Fx, Raw_Noise = Disturbance_generation_from_real_noise(
            fs=fs, Repet=0, wave_form=waveform, Pri_path=pri_path, Sec_path=sec_path
        )
        
        file_original = os.path.join(result_dir, '1_original.wav')
        save_audio_file(file_original, Dis, fs)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(Raw_Noise, np.ndarray):
            Raw_Noise_Tensor = torch.from_numpy(Raw_Noise).float()
        else:
            Raw_Noise_Tensor = Raw_Noise.float()
        
        input_slice = Raw_Noise_Tensor[:16000] if Raw_Noise_Tensor.shape[0] > 16000 else Raw_Noise_Tensor
        ai_input = input_slice.view(1, 1, -1).to(device)

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
            except: pass

        idx_res = 0
        if os.path.exists(model_resnet_path):
            try:
                res_net = CNNRes(channels=[[128], [128]*2], conv_kernels=[80, 3], 
                            conv_strides=[4, 1], conv_padding=[38, 1], pool_padding=[0, 0], num_classes=15).to(device)
                res_net.load_state_dict(torch.load(model_resnet_path, map_location=device))
                res_net.eval()
                with torch.no_grad():
                    idx_res = torch.argmax(res_net(ai_input)).item()
            except: pass

        step_size = 0.005
        w_init = np.zeros(1024)
        err_std = train_fxnlms_numpy(w_init, Fx, Dis, step_size=step_size)
        file_std = os.path.join(result_dir, '2_standard_nlms.wav')
        save_audio_file(file_std, err_std, fs)

        w_cnn = torch.from_numpy(pretrained_W[idx_cnn]).float()
        err_cnn = train_fxnlms_numpy(w_cnn, Fx, Dis, step_size=step_size)
        file_cnn = os.path.join(result_dir, '3_hybrid_cnn.wav')
        save_audio_file(file_cnn, err_cnn, fs)

        w_res = torch.from_numpy(pretrained_W[idx_res]).float()
        err_res = train_fxnlms_numpy(w_res, Fx, Dis, step_size=step_size)
        file_res = os.path.join(result_dir, '4_hybrid_resnet.wav')
        save_audio_file(file_res, err_res, fs)

        return {
            "fs": fs,
            "original": {"data": Dis.numpy(), "path": file_original},
            "std": {"data": err_std, "path": file_std},
            "cnn": {"data": err_cnn, "path": file_cnn, "idx": idx_cnn},
            "resnet": {"data": err_res, "path": file_res, "idx": idx_res}
        }

# --- GUI MAIN ---
class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Voice Processing System (ANC & ENC)")
        self.geometry("1250x900")
        
        # --- Variables ---
        self.current_file_path = None
        self.playing_file_path = None 
        self.recorder = AudioRecorder(os.path.join("input_audio", "record"))
        self.is_recording = False
        
        # Audio Player Object
        self.player = AudioPlayer()
        self.playback_paused = False

        # --- Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. Sidebar (File List)
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.setup_sidebar()

        # 2. Main Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, rowspan=2, padx=20, pady=(10, 0), sticky="nsew")
        self.tabview.add("ANC (Active Noise Cancellation)")
        self.tabview.add("ENC (Speech Enhancement)")

        self.setup_anc_tab()
        self.setup_enc_tab()

        # 3. Persistent Player
        self.player_frame = ctk.CTkFrame(self, height=140, corner_radius=10, fg_color="#1e1e1e")
        self.player_frame.grid(row=2, column=1, padx=20, pady=20, sticky="ew")
        self.setup_player_ui()

    def setup_sidebar(self):
        # Header
        ctk.CTkLabel(self.sidebar, text="AUDIO EXPLORER", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(30, 10))
        
        # Record Button (Nổi bật)
        self.btn_record = ctk.CTkButton(self.sidebar, text="● Start Recording", fg_color="#c0392b", hover_color="#e74c3c", 
                                        command=self.toggle_record, height=40, font=("Arial", 12, "bold"))
        self.btn_record.pack(pady=10, padx=20, fill="x")
        self.lbl_record_status = ctk.CTkLabel(self.sidebar, text="Ready", text_color="gray")
        self.lbl_record_status.pack(pady=(0, 10))

        ctk.CTkLabel(self.sidebar, text="Input Audio Files:", font=("Arial", 12, "bold"), anchor="w").pack(pady=5, padx=20, fill="x")

        # File List (Scrollable)
        self.file_list_frame = ctk.CTkScrollableFrame(self.sidebar, label_text="input_audio/")
        self.file_list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.refresh_file_list()

        # Button Refresh
        ctk.CTkButton(self.sidebar, text="↻ Refresh List", command=self.refresh_file_list, fg_color="transparent", border_width=1).pack(pady=10)

        # Selected File Info
        self.lbl_current_file = ctk.CTkLabel(self.sidebar, text="No file selected", wraplength=200, text_color="#2ecc71")
        self.lbl_current_file.pack(pady=20, padx=10)

    def refresh_file_list(self):
        # Xóa cũ
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
            
        root_dir = "input_audio"
        if not os.path.exists(root_dir): os.makedirs(root_dir)
        
        # Duyệt file đệ quy
        files_found = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith('.wav'):
                    full_path = os.path.join(dirpath, f)
                    # Tạo tên hiển thị (nếu trong subfolder thì hiện subfolder/file)
                    rel_path = os.path.relpath(full_path, root_dir)
                    files_found.append((rel_path, full_path))
        
        # Sắp xếp mới nhất lên đầu (nếu tên file có ngày tháng) hoặc theo tên
        files_found.sort(key=lambda x: x[0], reverse=True) 

        for display_name, full_path in files_found:
            btn = ctk.CTkButton(self.file_list_frame, text=f"🎵 {display_name}", 
                                anchor="w", fg_color="transparent", hover_color="#34495e",
                                command=lambda p=full_path: self.on_file_selected(p))
            btn.pack(fill="x", pady=2)

    def on_file_selected(self, path):
        self.current_file_path = path
        self.lbl_current_file.configure(text=f"SELECTED:\n{os.path.basename(path)}")
        self.load_to_player(path)

    # ================= RECORDING =================
    def toggle_record(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.configure(text="■ Stop Recording", fg_color="white", text_color="red")
            self.lbl_record_status.configure(text="Recording...", text_color="#e74c3c")
            self.recorder.start()
        else:
            self.is_recording = False
            self.btn_record.configure(text="● Start Recording", fg_color="#c0392b", text_color="white")
            self.lbl_record_status.configure(text="Saved", text_color="#2ecc71")
            filepath = self.recorder.stop()
            if filepath:
                messagebox.showinfo("Recording", f"Đã lưu file tại:\n{filepath}")
                self.refresh_file_list() # Cập nhật danh sách file
                self.on_file_selected(filepath) # Tự động chọn

    # ================= PLAYER UI =================
    def setup_player_ui(self):
        self.player_frame.grid_columnconfigure(1, weight=1)
        
        # Play/Pause Button
        self.btn_play_pause = ctk.CTkButton(self.player_frame, text="▶", width=50, height=50, 
                                            font=("Arial", 20), corner_radius=25,
                                            command=self.toggle_play)
        self.btn_play_pause.grid(row=0, column=0, rowspan=2, padx=20)
        
        # Info
        self.lbl_file_playing = ctk.CTkLabel(self.player_frame, text="Waiting for file...", font=("Arial", 12, "bold"), anchor="w")
        self.lbl_file_playing.grid(row=0, column=1, sticky="w", padx=10, pady=(10,0))

        self.lbl_time = ctk.CTkLabel(self.player_frame, text="00:00 / 00:00")
        self.lbl_time.grid(row=0, column=2, padx=20)

        # Waveform Canvas (Mini)
        self.fig_player, self.ax_player = plt.subplots(figsize=(8, 1.0), dpi=80)
        # SET MÀU NỀN CHO PLAYER
        self.fig_player.patch.set_facecolor('#1e1e1e')
        self.ax_player.set_facecolor('#1e1e1e')
        self.ax_player.axis('off')
        
        self.canvas_player = FigureCanvasTkAgg(self.fig_player, master=self.player_frame)
        self.canvas_player.get_tk_widget().grid(row=1, column=1, sticky="ew", padx=10)
        
        # Slider (Seek bar)
        self.slider_seek = ctk.CTkSlider(self.player_frame, from_=0, to=1, command=self.on_seek)
        self.slider_seek.grid(row=2, column=0, columnspan=3, sticky="ew", padx=20, pady=(5, 10))
        self.slider_seek.set(0)
        
        # Timer Loop
        self.update_player_ui_loop()

    def load_to_player(self, filepath):
        if not os.path.exists(filepath): return
        
        self.playing_file_path = filepath
        self.lbl_file_playing.configure(text=f"Playing: {os.path.basename(filepath)}")
        
        if self.player.load(filepath):
            # Reset UI
            self.btn_play_pause.configure(text="▶")
            self.slider_seek.configure(to=self.player.get_duration())
            self.slider_seek.set(0)
            self.lbl_time.configure(text=f"00:00 / {self.format_time(self.player.get_duration())}")
            
            # Vẽ Waveform
            try:
                sig = self.player.data
                # Downsample để vẽ nhanh
                step = max(1, len(sig) // 1000)
                sig_plot = sig[::step]
                
                self.ax_player.clear()
                self.ax_player.plot(sig_plot, color='#2cc985', linewidth=0.8)
                self.ax_player.axis('off')
                self.canvas_player.draw()
            except Exception as e: print(e)

    def toggle_play(self):
        if not self.playing_file_path: return
        
        if self.player.is_playing:
            self.player.pause()
            self.btn_play_pause.configure(text="▶")
        else:
            self.player.play()
            self.btn_play_pause.configure(text="⏸")

    def on_seek(self, value):
        if self.playing_file_path:
            ratio = value / self.player.get_duration()
            # self.player.seek cần tỉ lệ 0-1 hoặc index, 
            # nhưng ở đây slider max = duration, nên value/duration là sai logic class Player nếu class player nhận ratio
            # Class Player seek nhận ratio (0-1)
            self.player.seek(ratio)
            self.btn_play_pause.configure(text="⏸")

    def update_player_ui_loop(self):
        if self.player.is_playing:
            curr = self.player.get_current_time()
            dur = self.player.get_duration()
            
            # Update Slider (chỉ khi user không đang kéo giữ - khó detect trong ctk, nên ta update luôn)
            self.slider_seek.set(curr)
            self.lbl_time.configure(text=f"{self.format_time(curr)} / {self.format_time(dur)}")
            
            # Tự động chuyển icon khi hết bài
            if curr >= dur:
                self.btn_play_pause.configure(text="▶")
        
        self.after(100, self.update_player_ui_loop)

    def format_time(self, seconds):
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    
    def load_and_play_result(self, path):
        self.load_to_player(path)
        self.toggle_play()

    # ================= TAB ANC =================
    def setup_anc_tab(self):
        tab = self.tabview.tab("ANC (Active Noise Cancellation)")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Controls Frame
        ctrl_frame = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl_frame.grid(row=0, column=0, sticky="ew", pady=10)
        
        self.btn_run_anc = ctk.CTkButton(ctrl_frame, text="⚡ RUN ANC ALGORITHM", 
                                         height=40, font=("Arial", 14, "bold"),
                                         command=self.run_anc_process)
        self.btn_run_anc.pack(fill="x", padx=100)

        # Plot Area
        self.anc_fig = plt.figure(figsize=(10, 10), dpi=80)
        # SET MÀU NỀN TRƯỚC KHI VẼ
        self.anc_fig.patch.set_facecolor('#2b2b2b')
        self.anc_axs = self.anc_fig.subplots(4, 1, sharex=True)
        
        # Set style cho các axes trống
        for ax in self.anc_axs:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('white')
            
        self.anc_fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, hspace=0.4)
        
        self.anc_canvas = FigureCanvasTkAgg(self.anc_fig, master=tab)
        self.anc_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Result Buttons
        self.anc_play_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.anc_play_frame.grid(row=2, column=0, sticky="ew", pady=10)
        
        self.anc_btns = {}
        labels = ["Original", "Standard NLMS", "Hybrid CNN", "Hybrid ResNet"]
        for i, lbl in enumerate(labels):
            btn = ctk.CTkButton(self.anc_play_frame, text=f"Play {lbl}", state="disabled", width=120)
            btn.grid(row=0, column=i, padx=10, pady=5)
            self.anc_btns[lbl] = btn
        self.anc_play_frame.grid_columnconfigure((0,1,2,3), weight=1)

    def run_anc_process(self):
        if not self.current_file_path:
            messagebox.showwarning("Warning", "Vui lòng chọn file .wav từ danh sách bên trái!")
            return

        self.btn_run_anc.configure(state="disabled", text="Processing...")
        
        def thread_task():
            try:
                logic = ANCLogic()
                res = logic.run_anc(self.current_file_path)
                self.after(0, lambda: self.display_anc_results(res))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.btn_run_anc.configure(state="normal", text="⚡ RUN ANC ALGORITHM"))

        threading.Thread(target=thread_task).start()

    def display_anc_results(self, res):
        colors = ['white', 'orange', 'green', '#3498db']
        titles = [
            '1. Original Noise', 
            '2. Standard FxNLMS', 
            f'3. Hybrid CNN (Filter {res["cnn"]["idx"]})', 
            f'4. Hybrid ResNet (Filter {res["resnet"]["idx"]})'
        ]
        datas = [res['original']['data'], res['std']['data'], res['cnn']['data'], res['resnet']['data']]

        for i, ax in enumerate(self.anc_axs):
            ax.clear()
            ax.set_facecolor('#2b2b2b')
            # Downsample để vẽ nhanh
            plot_data = downsample_signal(datas[i])
            ax.plot(plot_data, color=colors[i], linewidth=0.8)
            
            ax.set_title(titles[i], color='white', fontsize=10, fontweight='bold')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            for spine in ax.spines.values(): spine.set_color('white')
            
            ax.autoscale(enable=True, axis='both', tight=True)
            ax.margins(y=0.1)

        self.anc_canvas.draw()

        # Update Buttons
        keys = ["Original", "Standard NLMS", "Hybrid CNN", "Hybrid ResNet"]
        paths = [res['original']['path'], res['std']['path'], res['cnn']['path'], res['resnet']['path']]
        
        msg = "ANC Completed!\nFiles:\n"
        for k, p in zip(keys, paths):
            self.anc_btns[k].configure(state="normal", command=lambda path=p: self.load_and_play_result(path))
            msg += f"- {k}\n"
        
        messagebox.showinfo("Success", msg)

    # ================= TAB ENC =================
    def setup_enc_tab(self):
        tab = self.tabview.tab("ENC (Speech Enhancement)")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1) # Plot expands

        # Control Row
        ctrl_frame = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl_frame.grid(row=0, column=0, sticky="ew", pady=10)

        ctk.CTkLabel(ctrl_frame, text="Noise Type:", font=("Arial", 12)).pack(side="left", padx=10)
        self.noise_options = {
            "Household Appliance": "Household_Appliance",
            "TV / Radio": "TVnRadio",
            "Vehicles": "Vechicles",
            "Verbal Human": "Verbal_Human"
        }
        self.combo_noise = ctk.CTkComboBox(ctrl_frame, values=list(self.noise_options.keys()), width=200)
        self.combo_noise.set("Household Appliance")
        self.combo_noise.pack(side="left", padx=10)

        self.btn_run_enc = ctk.CTkButton(ctrl_frame, text="⚡ RUN DENOISE", 
                                         height=40, font=("Arial", 14, "bold"), fg_color="#8e44ad",
                                         command=self.run_enc_process)
        self.btn_run_enc.pack(side="left", fill="x", expand=True, padx=20)

        # Plot Area
        self.enc_fig = plt.figure(figsize=(8, 6), dpi=80)
        self.enc_fig.patch.set_facecolor('#2b2b2b')
        self.enc_axs = self.enc_fig.subplots(2, 1, sharex=True)
        
        # Set Style Empty Plots
        for ax in self.enc_axs:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('white')

        self.enc_fig.tight_layout(pad=3)
        self.enc_canvas = FigureCanvasTkAgg(self.enc_fig, master=tab)
        self.enc_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Buttons
        self.enc_play_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.enc_play_frame.grid(row=2, column=0, sticky="ew", pady=10)
        
        self.btn_enc_orig = ctk.CTkButton(self.enc_play_frame, text="Play Original", state="disabled", width=150)
        self.btn_enc_orig.pack(side="left", padx=20, expand=True)
        
        self.btn_enc_clean = ctk.CTkButton(self.enc_play_frame, text="Play Cleaned", state="disabled", fg_color="#2ecc71", width=150)
        self.btn_enc_clean.pack(side="left", padx=20, expand=True)

    def run_enc_process(self):
        if not self.current_file_path:
            messagebox.showwarning("Warning", "Vui lòng chọn file .wav trước!")
            return

        selected_key = self.combo_noise.get()
        noise_class = self.noise_options[selected_key]
        self.btn_run_enc.configure(state="disabled", text="Loading Model & Processing...")
        
        def thread_task():
            try:
                model_name = f"DDAE_FC_{noise_class}.h5"
                model_path = os.path.join("data", "enc", model_name)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Missing model: {model_path}")

                model = load_model(model_path, compile=False)
                audio = data_tools.audio_files_to_numpy(self.current_file_path)
                os.makedirs('./temp_split/gui_images/', exist_ok=True)
                
                segments = data_tools.split_into_one_second(audio, './temp_split/', 'gui', False)
                segments_array = np.array(segments)

                mag_db, phase = data_tools.numpy_audio_to_matrix_spectrogram(segments_array, './temp_split/gui_images/')
                X_in = data_tools.scaled_in(mag_db)
                
                X_pred = model.predict(X_in)
                inv_sca_X_pred = data_tools.inv_scaled_ou(X_pred)
                X_denoise = mag_db - inv_sca_X_pred

                audio_reconstruct = data_tools.matrix_spectrogram_to_numpy_audio(
                    X_denoise, phase, segments_array.shape[1], './temp_split/gui_images/')

                audio_flat = audio_reconstruct.flatten()
                peak = np.max(np.abs(audio_flat))
                if peak > 0: audio_flat = (audio_flat / peak * 0.8)

                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join("result", "enc")
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"Denoised_{now}.wav")
                sf.write(out_path, audio_flat, config_params.SAMPLE_RATE, 'PCM_24')
                
                # Đọc file gốc để vẽ
                sig_orig, _ = sf.read(self.current_file_path)

                self.after(0, lambda: self.display_enc_results(sig_orig, audio_flat, out_path))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error ENC", str(e)))
            finally:
                self.after(0, lambda: self.btn_run_enc.configure(state="normal", text="⚡ RUN DENOISE"))

        threading.Thread(target=thread_task).start()

    def display_enc_results(self, original, cleaned, clean_path):
        # DOWNSAMPLE để tránh lỗi vẽ nếu file quá dài
        original = downsample_signal(original)
        cleaned = downsample_signal(cleaned)
        
        # Plot 1
        ax1 = self.enc_axs[0]
        ax1.clear()
        ax1.set_facecolor('#2b2b2b')
        ax1.plot(original, color='#3498db', linewidth=0.5)
        ax1.set_title("Original (Noisy)", color='white')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values(): spine.set_color('white')

        # Plot 2
        ax2 = self.enc_axs[1]
        ax2.clear()
        ax2.set_facecolor('#2b2b2b')
        ax2.plot(cleaned, color='#2ecc71', linewidth=0.5)
        ax2.set_title("Result (Cleaned)", color='white')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values(): spine.set_color('white')

        self.enc_canvas.draw()
        messagebox.showinfo("ENC Completed", f"Đã lọc xong!\nFile lưu tại: {clean_path}")

        self.btn_enc_orig.configure(state="normal", command=lambda: self.load_and_play_result(self.current_file_path))
        self.btn_enc_clean.configure(state="normal", command=lambda: self.load_and_play_result(clean_path))

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
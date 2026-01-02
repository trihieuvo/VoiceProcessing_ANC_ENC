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

# --- IMPORT MODULES ---
from src.anc.network import CNN, CNNRes
from src.anc.utils import load_audio_file, load_anc_paths, load_pretrained_filters, save_audio_file
from src.anc.simulation import Disturbance_generation_from_real_noise
from src.enc import data_tools
from src.enc import config_params

# Cấu hình giao diện
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")
PLOT_BG_COLOR = '#2b2b2b'

# --- HÀM TỐI ƯU HÓA ---
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
    """Giảm số lượng điểm vẽ"""
    if len(signal) > max_points:
        step = len(signal) // max_points
        return signal[::step]
    return signal


# --- CLASS AUDIO PLAYER ---
class AudioPlayer:
    def __init__(self):
        self.data = None
        self.fs = 16000
        self.current_idx = 0
        self.is_playing = False
        self._start_time = 0
        self._start_idx = 0
        
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
        self.stop() 
        self.is_playing = True
        data_to_play = self.data[self.current_idx:]
        sd.play(data_to_play, self.fs)
        self._start_time = time.time()
        self._start_idx = self.current_idx

    def pause(self):
        if self.is_playing:
            sd.stop()
            self.is_playing = False
            elapsed = time.time() - self._start_time
            self.current_idx = self._start_idx + int(elapsed * self.fs)
            if self.current_idx > len(self.data):
                self.current_idx = 0

    def stop(self):
        sd.stop()
        self.is_playing = False

    def seek(self, ratio):
        if self.data is None: return
        self.current_idx = int(len(self.data) * ratio)
        if self.is_playing:
            self.play()

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

        self.title("ANC & ENC Audio Processing Application")
        self.geometry("1280x900")
        
        self.current_file_path = None
        self.recorder = AudioRecorder(os.path.join("input_audio", "record"))
        self.is_recording = False
        self.player = AudioPlayer()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. Sidebar
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.setup_sidebar()

        # 2. Main Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, rowspan=2, padx=20, pady=(10, 0), sticky="nsew")
        self.tabview.add("ANC Mode")
        self.tabview.add("ENC Mode")

        self.setup_anc_tab()
        self.setup_enc_tab()

        # 3. Persistent Player
        self.player_frame = ctk.CTkFrame(self, height=120, corner_radius=0, fg_color="#1a1a1a")
        self.player_frame.grid(row=2, column=1, padx=0, pady=0, sticky="ew")
        self.setup_player_ui()

    def setup_sidebar(self):
        ctk.CTkLabel(self.sidebar, text="INPUT FILES:", font=("Arial", 16, "bold"), text_color="gray").pack(pady=(20, 10), padx=20, anchor="w")
        
        self.btn_record = ctk.CTkButton(self.sidebar, text="RECORD AUDIO", fg_color="#c0392b", hover_color="#e74c3c", 
                                        command=self.toggle_record, height=35, font=("Arial", 12, "bold"))
        self.btn_record.pack(pady=5, padx=20, fill="x")
        self.lbl_record_status = ctk.CTkLabel(self.sidebar, text="Ready to record", text_color="gray", font=("Arial", 11))
        self.lbl_record_status.pack(pady=(0, 10))

        self.file_list_frame = ctk.CTkScrollableFrame(self.sidebar, label_text="Available Audio", fg_color="transparent")
        self.file_list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.refresh_file_list()

        ctk.CTkButton(self.sidebar, text="Refresh List", command=self.refresh_file_list, 
                      fg_color="transparent", border_width=1, text_color="gray", height=25).pack(pady=10)

        ctk.CTkLabel(self.sidebar, text="SELECTED FILE:", font=("Arial", 12, "bold"), text_color="gray").pack(padx=20, anchor="w")
        self.lbl_current_file = ctk.CTkLabel(self.sidebar, text="None", wraplength=220, text_color="#2ecc71", font=("Arial", 12))
        self.lbl_current_file.pack(pady=(0, 20), padx=20, anchor="w")

    def refresh_file_list(self):
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
            
        root_dir = "input_audio"
        if not os.path.exists(root_dir): os.makedirs(root_dir)
        
        files_found = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith('.wav'):
                    full_path = os.path.join(dirpath, f)
                    rel_path = os.path.relpath(full_path, root_dir)
                    files_found.append((rel_path, full_path))
        
        files_found.sort(key=lambda x: x[0], reverse=True) 

        for display_name, full_path in files_found:
            btn = ctk.CTkButton(self.file_list_frame, text=f"  {display_name}", 
                                anchor="w", fg_color="transparent", text_color="#ecf0f1", hover_color="#34495e",
                                height=30, command=lambda p=full_path: self.on_file_selected(p))
            btn.pack(fill="x", pady=1)

    def on_file_selected(self, path):
        self.current_file_path = path
        self.lbl_current_file.configure(text=os.path.basename(path))
        self.load_to_player(path)

    def toggle_record(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.configure(text="STOP RECORDING", fg_color="#e74c3c")
            self.lbl_record_status.configure(text="Recording in progress...", text_color="#e74c3c")
            self.recorder.start()
        else:
            self.is_recording = False
            self.btn_record.configure(text="RECORD AUDIO", fg_color="#c0392b")
            self.lbl_record_status.configure(text="File saved successfully", text_color="#2ecc71")
            filepath = self.recorder.stop()
            if filepath:
                self.refresh_file_list()
                self.on_file_selected(filepath)

    def setup_player_ui(self):
        self.player_frame.grid_columnconfigure(1, weight=1)
        
        self.btn_play_pause = ctk.CTkButton(self.player_frame, text="PLAY", width=80, height=40, 
                                            font=("Arial", 12, "bold"), fg_color="#27ae60", hover_color="#2ecc71",
                                            command=self.toggle_play)
        self.btn_play_pause.grid(row=0, column=0, rowspan=2, padx=20)
        
        self.lbl_file_playing = ctk.CTkLabel(self.player_frame, text="No Audio Loaded", font=("Arial", 12, "bold"), anchor="w")
        self.lbl_file_playing.grid(row=0, column=1, sticky="w", padx=10, pady=(15,0))

        self.lbl_time = ctk.CTkLabel(self.player_frame, text="00:00 / 00:00", font=("Arial", 12))
        self.lbl_time.grid(row=0, column=2, padx=20)

        self.fig_player, self.ax_player = plt.subplots(figsize=(8, 0.8), dpi=80)
        self.fig_player.patch.set_facecolor('#1a1a1a')
        self.ax_player.set_facecolor('#1a1a1a')
        self.ax_player.axis('off')
        
        self.canvas_player = FigureCanvasTkAgg(self.fig_player, master=self.player_frame)
        self.canvas_player.get_tk_widget().grid(row=1, column=1, sticky="ew", padx=10)
        
        self.slider_seek = ctk.CTkSlider(self.player_frame, from_=0, to=1, command=self.on_seek, height=15)
        self.slider_seek.grid(row=2, column=0, columnspan=3, sticky="ew", padx=20, pady=(5, 15))
        self.slider_seek.set(0)
        
        self.update_player_ui_loop()

    def load_to_player(self, filepath):
        if not os.path.exists(filepath): return
        
        self.lbl_file_playing.configure(text=f"Playing: {os.path.basename(filepath)}")
        
        if self.player.load(filepath):
            self.btn_play_pause.configure(text="PLAY")
            self.slider_seek.configure(to=self.player.get_duration())
            self.slider_seek.set(0)
            self.lbl_time.configure(text=f"00:00 / {self.format_time(self.player.get_duration())}")
            
            try:
                sig = self.player.data
                step = max(1, len(sig) // 1000)
                sig_plot = sig[::step]
                
                self.ax_player.clear()
                self.ax_player.plot(sig_plot, color='#2cc985', linewidth=0.8)
                self.ax_player.axis('off')
                self.canvas_player.draw()
            except Exception: pass

    def toggle_play(self):
        if self.player.is_playing:
            self.player.pause()
            self.btn_play_pause.configure(text="PLAY", fg_color="#27ae60")
        else:
            self.player.play()
            self.btn_play_pause.configure(text="PAUSE", fg_color="#d35400")

    def on_seek(self, value):
        ratio = value / self.player.get_duration() if self.player.get_duration() > 0 else 0
        self.player.seek(ratio)
        if self.player.is_playing:
             self.btn_play_pause.configure(text="PAUSE")

    def update_player_ui_loop(self):
        if self.player.is_playing:
            curr = self.player.get_current_time()
            dur = self.player.get_duration()
            self.slider_seek.set(curr)
            self.lbl_time.configure(text=f"{self.format_time(curr)} / {self.format_time(dur)}")
            if curr >= dur:
                self.btn_play_pause.configure(text="PLAY", fg_color="#27ae60")
        self.after(100, self.update_player_ui_loop)

    def format_time(self, seconds):
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
    
    def load_and_play_result(self, path):
        self.load_to_player(path)
        self.player.play()
        self.btn_play_pause.configure(text="PAUSE", fg_color="#d35400")

    # ================= TAB ANC =================
    def setup_anc_tab(self):
        tab = self.tabview.tab("ANC Mode")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        self.btn_run_anc = ctk.CTkButton(tab, text="RUN ANC SIMULATION", 
                                         height=40, font=("Arial", 13, "bold"), fg_color="#2980b9",
                                         command=self.run_anc_process)
        self.btn_run_anc.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

        self.anc_fig = plt.figure(figsize=(10, 10), dpi=80)
        self.anc_fig.patch.set_facecolor(PLOT_BG_COLOR)
        self.anc_axs = self.anc_fig.subplots(4, 1, sharex=True)
        
        for ax in self.anc_axs:
            ax.set_facecolor(PLOT_BG_COLOR)
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values(): spine.set_color('white')
            
        self.anc_fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, hspace=0.4)
        
        self.anc_canvas = FigureCanvasTkAgg(self.anc_fig, master=tab)
        self.anc_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        self.anc_play_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.anc_play_frame.grid(row=2, column=0, sticky="ew", pady=10)
        
        self.anc_btns = {}
        labels = ["Original", "Standard NLMS", "Hybrid CNN", "Hybrid ResNet"]
        for i, lbl in enumerate(labels):
            btn = ctk.CTkButton(self.anc_play_frame, text=f"Listen: {lbl}", state="disabled", width=140, height=30)
            btn.grid(row=0, column=i, padx=5, pady=5)
            self.anc_btns[lbl] = btn
        self.anc_play_frame.grid_columnconfigure((0,1,2,3), weight=1)

    def run_anc_process(self):
        if not self.current_file_path:
            messagebox.showwarning("Warning", "Please select a file first!")
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
                self.after(0, lambda: self.btn_run_anc.configure(state="normal", text="RUN ANC SIMULATION"))

        threading.Thread(target=thread_task).start()

    def display_anc_results(self, res):
        colors = ['white', 'orange', 'green', '#3498db']
        titles = ['Original Noise', 'Standard FxNLMS', 
                  f'Hybrid CNN (Filter {res["cnn"]["idx"]})', f'Hybrid ResNet (Filter {res["resnet"]["idx"]})']
        datas = [res['original']['data'], res['std']['data'], res['cnn']['data'], res['resnet']['data']]

        for i, ax in enumerate(self.anc_axs):
            ax.clear()
            ax.set_facecolor(PLOT_BG_COLOR)
            plot_data = downsample_signal(datas[i])
            ax.plot(plot_data, color=colors[i], linewidth=0.8)
            ax.set_title(titles[i], color='white', fontsize=9, fontweight='bold')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            ax.autoscale(enable=True, axis='both', tight=True)
            ax.margins(y=0.1)

        self.anc_canvas.draw()

        keys = ["Original", "Standard NLMS", "Hybrid CNN", "Hybrid ResNet"]
        paths = [res['original']['path'], res['std']['path'], res['cnn']['path'], res['resnet']['path']]
        for k, p in zip(keys, paths):
            self.anc_btns[k].configure(state="normal", fg_color="#34495e", command=lambda path=p: self.load_and_play_result(path))
        
        messagebox.showinfo("Success", "ANC Simulation Completed!")

    # ================= TAB ENC =================
    def setup_enc_tab(self):
        tab = self.tabview.tab("ENC Mode")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ctrl_frame = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl_frame.grid(row=0, column=0, sticky="ew", pady=10)

        ctk.CTkLabel(ctrl_frame, text="Noise Type:").pack(side="left", padx=10)
        self.noise_options = {"Household Appliance": "Household_Appliance", "TV / Radio": "TVnRadio", 
                              "Vehicles": "Vechicles", "Verbal Human": "Verbal_Human"}
        self.combo_noise = ctk.CTkComboBox(ctrl_frame, values=list(self.noise_options.keys()), width=200)
        self.combo_noise.set("Household Appliance")
        self.combo_noise.pack(side="left", padx=10)

        self.btn_run_enc = ctk.CTkButton(ctrl_frame, text="RUN ENC SIMULATION", height=40, 
                                         font=("Arial", 13, "bold"), fg_color="#8e44ad",
                                         command=self.run_enc_process)
        self.btn_run_enc.pack(side="left", fill="x", expand=True, padx=20)

        self.enc_fig = plt.figure(figsize=(8, 6), dpi=80)
        self.enc_fig.patch.set_facecolor(PLOT_BG_COLOR)
        self.enc_axs = self.enc_fig.subplots(2, 1, sharex=True)
        
        for ax in self.enc_axs:
            ax.set_facecolor(PLOT_BG_COLOR)
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values(): spine.set_color('white')

        # Dùng logic subplots_adjust 
        self.enc_fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, hspace=0.3)
        
        self.enc_canvas = FigureCanvasTkAgg(self.enc_fig, master=tab)
        self.enc_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        self.enc_play_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.enc_play_frame.grid(row=2, column=0, sticky="ew", pady=10)
        
        self.btn_enc_orig = ctk.CTkButton(self.enc_play_frame, text="Listen: Original", state="disabled", width=180, height=30)
        self.btn_enc_orig.pack(side="left", padx=20, expand=True)
        
        self.btn_enc_clean = ctk.CTkButton(self.enc_play_frame, text="Listen: Cleaned", state="disabled", width=180, height=30)
        self.btn_enc_clean.pack(side="left", padx=20, expand=True)

    def run_enc_process(self):
        if not self.current_file_path:
            messagebox.showwarning("Warning", "Please select a file first!")
            return

        selected_key = self.combo_noise.get()
        noise_class = self.noise_options[selected_key]
        self.btn_run_enc.configure(state="disabled", text="Processing...")
        
        def thread_task():
            try:
                model_name = f"DDAE_FC_{noise_class}.h5"
                model_path = os.path.join("data", "enc", model_name)
                if not os.path.exists(model_path): raise FileNotFoundError(f"Missing model: {model_path}")

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
                
                sig_orig, fs_orig = sf.read(self.current_file_path)
                
                self.after(0, lambda: self.display_enc_results(sig_orig, audio_flat, out_path, fs_orig))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error ENC", str(e)))
            finally:
                self.after(0, lambda: self.btn_run_enc.configure(state="normal", text="RUN DENOISE"))

        threading.Thread(target=thread_task).start()

    def display_enc_results(self, original, cleaned, clean_path, fs_orig):
        # Chuyển Stereo -> Mono 
        if len(original.shape) > 1: original = original[:, 0]
        
        # Tính thời gian dài tín hiệu
        dur_orig = len(original) / fs_orig
        dur_clean = len(cleaned) / config_params.SAMPLE_RATE

        # Downsample
        orig_ds = downsample_signal(original)
        clean_ds = downsample_signal(cleaned)

        # Tạo trục thời gian (Time Axis)
        t_orig = np.linspace(0, dur_orig, len(orig_ds))
        t_clean = np.linspace(0, dur_clean, len(clean_ds))
        
        # Plot 1 (Original)
        ax1 = self.enc_axs[0]
        ax1.clear()
        ax1.set_facecolor(PLOT_BG_COLOR)
        # Vẽ theo trục thời gian
        ax1.plot(t_orig, orig_ds, color='#3498db', linewidth=0.5)
        ax1.set_title("Original (Noisy)", color='white', fontsize=20, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')
        ax1.autoscale(enable=True, axis='both', tight=True)
        ax1.margins(y=0.1)

        # Plot 2 (Cleaned)
        ax2 = self.enc_axs[1]
        ax2.clear()
        ax2.set_facecolor(PLOT_BG_COLOR)
        # Vẽ theo trục thời gian
        ax2.plot(t_clean, clean_ds, color='#2ecc71', linewidth=0.5)
        ax2.set_title("Result (Cleaned)", color='white', fontsize=20, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(colors='white')
        ax2.autoscale(enable=True, axis='both', tight=True)
        ax2.margins(y=0.1)

        self.enc_canvas.draw()
        messagebox.showinfo("ENC Completed", f"Denoising Successful!\nSaved at: {clean_path}")

        self.btn_enc_orig.configure(state="normal", fg_color="#34495e", command=lambda: self.load_and_play_result(self.current_file_path))
        self.btn_enc_clean.configure(state="normal", fg_color="#27ae60", command=lambda: self.load_and_play_result(clean_path))

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
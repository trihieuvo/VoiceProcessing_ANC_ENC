import os
import warnings
import csv

# --- 1. CẤU HÌNH ẨN CẢNH BÁO RÁC ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")
PLOT_BG_COLOR = '#2b2b2b'
class DataLogic:
    def __init__(self):
        self.filters_data = None
        self.pri_path = None
        self.sec_path = None
        self.filter_key_name = "Unknown" 
        
        # Đường dẫn file
        self.filter_mat_path = os.path.join('data', 'anc', 'Pretrained_Control_filters.mat')
        self.pri_mat_path = os.path.join('data', 'anc', 'Primary_path.mat')
        self.sec_mat_path = os.path.join('data', 'anc', 'Secondary_path.mat')

    def load_filters(self):
        if not os.path.exists(self.filter_mat_path):
            raise FileNotFoundError(f"File not found: {self.filter_mat_path}")
        
        data = sio.loadmat(self.filter_mat_path)

        key = 'Wc' if 'Wc' in data else ([k for k in data.keys() if not k.startswith('__')][0])
        self.filter_key_name = key
        self.filters_data = data[key] 
        return self.filters_data

    def load_paths(self):
        if not os.path.exists(self.pri_mat_path) or not os.path.exists(self.sec_mat_path):
            raise FileNotFoundError("Primary or Secondary path .mat files missing.")

        pri_data = sio.loadmat(self.pri_mat_path)
        sec_data = sio.loadmat(self.sec_mat_path)

        p_key = [k for k in pri_data.keys() if not k.startswith('__')][0]
        s_key = [k for k in sec_data.keys() if not k.startswith('__')][0]

        self.pri_path = pri_data[p_key].flatten()
        self.sec_path = sec_data[s_key].flatten()
        
        return self.pri_path, self.sec_path

    def export_csv(self, data, default_name):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path: return None

        try:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                if data.ndim == 1: 
                    writer.writerow(["Index", "Value"])
                    for i, val in enumerate(data):
                        writer.writerow([i, val])
                else: 
                    header = ["Index"] + [f"Filter_{i+1}" for i in range(data.shape[0])]
                    writer.writerow(header)
                    length = data.shape[1]
                    for i in range(length):
                        row = [i] + list(data[:, i])
                        writer.writerow(row)
            return file_path
        except Exception as e:
            raise e

class ViewDataApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ANC Data Inspector")
        self.geometry("1200x850")
        self.logic = DataLogic()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.tabview.add("Control Filters")
        self.tabview.add("Acoustic Paths")

        self.setup_filters_tab()
        self.setup_paths_tab()

    # ================= TAB 1: FILTERS =================
    def setup_filters_tab(self):
        tab = self.tabview.tab("Control Filters")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        self.fig_filters = plt.figure(figsize=(12, 10), dpi=80)
        self.fig_filters.patch.set_facecolor(PLOT_BG_COLOR)
        
        self.axs_filters = self.fig_filters.subplots(5, 3) 
        
        for ax in self.axs_filters.flat:
            ax.set_facecolor(PLOT_BG_COLOR)
            ax.tick_params(colors='white', labelsize=6)
            for spine in ax.spines.values(): spine.set_color('white')
            ax.grid(False)

        self.fig_filters.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.6, wspace=0.3)
        
        self.canvas_filters = FigureCanvasTkAgg(self.fig_filters, master=tab)
        self.canvas_filters.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctrl_frame = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl_frame.grid(row=1, column=0, sticky="ew", pady=10)

        ctk.CTkButton(ctrl_frame, text="LOAD FILTERS", command=self.load_and_plot_filters, 
                      height=40, font=("Arial", 12, "bold")).pack(side="left", padx=20, expand=True)
        
        self.btn_export_filters = ctk.CTkButton(ctrl_frame, text="EXPORT CSV", command=self.export_filters,
                                                state="disabled", fg_color="#d35400", hover_color="#e67e22",
                                                height=40, font=("Arial", 12, "bold"))
        self.btn_export_filters.pack(side="left", padx=20, expand=True)

    def load_and_plot_filters(self):
        try:
            filters = self.logic.load_filters()
            num_filters = filters.shape[0]
            y_min, y_max = np.min(filters), np.max(filters)
            
            flat_axs = self.axs_filters.flatten()
            
            for i, ax in enumerate(flat_axs):
                ax.clear()
                ax.set_facecolor(PLOT_BG_COLOR)
                
                if i < num_filters:
                    ax.plot(filters[i], color='#3498db', linewidth=0.8)
                    # [UPDATE] Hiển thị tên Filter rõ ràng hơn
                    ax.set_title(f"Filter Index: {i} (Label {i})", color='white', fontsize=9, fontweight='bold')
                    ax.set_ylim([y_min, y_max])
                    ax.grid(True, alpha=0.3)
                else:
                    ax.axis('off')
                
                ax.tick_params(colors='white', labelsize=6)
                for spine in ax.spines.values(): spine.set_color('white')

            self.canvas_filters.draw()
            self.btn_export_filters.configure(state="normal")
            # Hiển thị tên biến lấy từ file mat
            messagebox.showinfo("Loaded", f"Loaded {num_filters} filters from variable '{self.logic.filter_key_name}'")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_filters(self):
        try:
            path = self.logic.export_csv(self.logic.filters_data, "view_filters.csv")
            if path:
                messagebox.showinfo("Success", f"Filters saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # ================= TAB 2: PATHS =================
    def setup_paths_tab(self):
        tab = self.tabview.tab("Acoustic Paths")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        self.fig_paths = plt.figure(figsize=(10, 6), dpi=80)
        self.fig_paths.patch.set_facecolor(PLOT_BG_COLOR)
        
        # [UPDATE] sharex=False để trục X độc lập
        self.axs_paths = self.fig_paths.subplots(2, 1, sharex=False)
        
        for ax in self.axs_paths:
            ax.set_facecolor(PLOT_BG_COLOR)
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values(): spine.set_color('white')

        self.fig_paths.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1, hspace=0.4)
        
        self.canvas_paths = FigureCanvasTkAgg(self.fig_paths, master=tab)
        self.canvas_paths.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctrl_frame = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl_frame.grid(row=1, column=0, sticky="ew", pady=10)

        ctk.CTkButton(ctrl_frame, text="LOAD PATHS", command=self.load_and_plot_paths,
                      height=40, font=("Arial", 12, "bold")).pack(side="left", padx=20, expand=True)

        self.btn_export_pri = ctk.CTkButton(ctrl_frame, text="EXPORT PRIMARY", command=lambda: self.export_path('pri'),
                                            state="disabled", fg_color="#2980b9", width=150, height=40)
        self.btn_export_pri.pack(side="left", padx=10)

        self.btn_export_sec = ctk.CTkButton(ctrl_frame, text="EXPORT SECONDARY", command=lambda: self.export_path('sec'),
                                            state="disabled", fg_color="#8e44ad", width=150, height=40)
        self.btn_export_sec.pack(side="left", padx=10)

    def load_and_plot_paths(self):
        try:
            pri, sec = self.logic.load_paths()
            
            # Plot Primary
            ax1 = self.axs_paths[0]
            ax1.clear()
            ax1.set_facecolor(PLOT_BG_COLOR)
            ax1.plot(pri, color='#3498db', linewidth=1)
            ax1.set_title("Primary Path Impulse Response", color='white', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(colors='white', labelsize=8)
            for spine in ax1.spines.values(): spine.set_color('white')
            
            # Plot Secondary
            ax2 = self.axs_paths[1]
            ax2.clear()
            ax2.set_facecolor(PLOT_BG_COLOR)
            ax2.plot(sec, color='#e74c3c', linewidth=1)
            ax2.set_title("Secondary Path Impulse Response", color='white', fontweight='bold')
            ax2.set_xlabel("Samples", color='white') # Chỉ hiện nhãn trục X ở biểu đồ dưới
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(colors='white', labelsize=8)
            for spine in ax2.spines.values(): spine.set_color('white')
            
            self.canvas_paths.draw()
            self.btn_export_pri.configure(state="normal")
            self.btn_export_sec.configure(state="normal")
            messagebox.showinfo("Loaded", "Primary and Secondary paths loaded.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_path(self, p_type):
        try:
            if p_type == 'pri':
                data = self.logic.pri_path
                name = "view_primary_path.csv"
            else:
                data = self.logic.sec_path
                name = "view_secondary_path.csv"
                
            path = self.logic.export_csv(data, name)
            if path:
                messagebox.showinfo("Success", f"Path saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

if __name__ == "__main__":
    app = ViewDataApp()
    app.mainloop()
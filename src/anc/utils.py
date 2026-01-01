import os
import torch
import torchaudio.transforms as T
import scipy.io as sio
import numpy as np
import soundfile as sf  # <--- Dùng thư viện này thay cho torchaudio.load

# Cấu hình hằng số
FS = 16000

def load_audio_file(filepath, target_fs=FS):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
    try:
        # data: numpy array [Frames, Channels]
        data, sample_rate = sf.read(filepath)
    except Exception as e:
        raise RuntimeError(f"Lỗi khi đọc file wav bằng soundfile: {e}")


    waveform = torch.from_numpy(data).float()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()

    
    # Resample
    if sample_rate != target_fs:
        resampler = T.Resample(sample_rate, target_fs, dtype=waveform.dtype)
        waveform = resampler(waveform)
    
    # lấy kênh đầu tiên nếu là stereo
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]
        
    return waveform, target_fs

def load_anc_paths(pri_path_file, sec_path_file):
    """Tải Primary và Secondary paths từ file .mat"""
    if not os.path.exists(pri_path_file) or not os.path.exists(sec_path_file):
        raise FileNotFoundError("Không tìm thấy file .mat đường truyền")

    pri_data = sio.loadmat(pri_path_file)
    sec_data = sio.loadmat(sec_path_file)

    # Lấy dữ liệu và ép về mảng 1 chiều
    pri_path = pri_data.get('Pz1', pri_data.get('Primary_path')).squeeze()
    sec_path = sec_data.get('S', sec_data.get('Secondary_path')).squeeze()

    return pri_path, sec_path

def load_pretrained_filters(filepath):
    """Tải bộ lọc đã train trước (Wc)"""
    if not os.path.exists(filepath):
         raise FileNotFoundError(f"Không tìm thấy file filter: {filepath}")
         
    data = sio.loadmat(filepath)
    if 'Wc' in data:
        return data['Wc']
    elif 'Control_filter' in data:
        return data['Control_filter']
    else:
        keys = [k for k in data.keys() if not k.startswith('__')]
        return data[keys[0]]
    
def save_audio_file(filepath, data, sample_rate=16000):
    """
    Lưu dữ liệu âm thanh vào file .wav
    Args:
        filepath: Đường dẫn lưu file
        data: Dữ liệu âm thanh (Tensor hoặc Numpy array)
        sample_rate: Tần số lấy mẫu
    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    data = np.squeeze(data)
    
    # Lưu file
    sf.write(filepath, data, sample_rate)
    print(f"Đã lưu file âm thanh: {filepath}")
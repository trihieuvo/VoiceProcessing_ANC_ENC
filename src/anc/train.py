import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import soundfile as sf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast # <--- Thư viện tăng tốc FP16

# Import kiến trúc mạng
from network import CNN

# --- CẤU HÌNH TỐI ƯU CHO RTX 3050 ---
BATCH_SIZE = 256        # Tăng lên 128 (hoặc 256 nếu VRAM chịu nổi) để tận dụng GPU
LEARNING_RATE = 0.001
EPOCHS = 50
TARGET_FS = 16000
NUM_CLASSES = 15

# Cấu hình phần cứng
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Bật chế độ tìm thuật toán nhanh nhất cho CNN
torch.backends.cudnn.benchmark = True 

# Số nhân CPU dùng để load dữ liệu (RTX 3050 thường đi với CPU 6-8 nhân, set 4 là an toàn)
NUM_WORKERS = 4 

# ĐƯỜNG DẪN DATASET
DATASET_ROOT = r'D:\UTE\VoiceProcessing\project\DATA\Processed_Dataset_For_Train'

# --- 1. DATASET CLASS ---
class ANCDataset(Dataset):
    def __init__(self, root_dir, subset, target_fs=16000):
        self.root_dir = os.path.join(root_dir, subset)
        self.files = []
        self.target_fs = target_fs
        
        for label in range(NUM_CLASSES):
            folder_path = os.path.join(self.root_dir, str(label))
            if os.path.exists(folder_path):
                # Liệt kê file nhanh hơn bằng os.scandir
                with os.scandir(folder_path) as entries:
                    for entry in entries:
                        if entry.name.endswith('.wav'):
                            self.files.append((entry.path, label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        try:
            # Dùng soundfile đọc cực nhanh
            audio_data, fs = sf.read(path, dtype='float32') # Đọc trực tiếp float32
            
            waveform = torch.from_numpy(audio_data)
            
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0) 
            else:
                waveform = waveform.t()

            # Resample (Chỉ làm nếu cần thiết để tiết kiệm CPU)
            if fs != self.target_fs:
                resampler = torchaudio.transforms.Resample(fs, self.target_fs)
                waveform = resampler(waveform)
            
            # Cắt/Pad nhanh
            max_len = 16000
            curr_len = waveform.shape[1]
            if curr_len > max_len:
                waveform = waveform[:, :max_len]
            elif curr_len < max_len:
                waveform = torch.nn.functional.pad(waveform, (0, max_len - curr_len))
            
            return waveform, label
        except Exception as e:
            # print(f"Lỗi: {path}") 
            return torch.zeros(1, 16000), label

# --- 2. MODEL ---
def get_model():
    model = CNN(
        channels=[[16], [32], [64], [128]], 
        conv_kernels=[7, 5, 3, 3],
        conv_strides=[1, 1, 1, 1],
        conv_padding=[3, 2, 1, 1],
        pool_padding=[0, 0, 0, 0],
        num_classes=NUM_CLASSES
    )
    return model.to(DEVICE)

# --- 3. TRAINING LOOP (CÓ AMP) ---
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    loop = tqdm(loader, leave=False)
    for data, targets in loop:
        # non_blocking=True giúp chuyển dữ liệu song song với tính toán
        data = data.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        # --- TĂNG TỐC VỚI MIXED PRECISION ---
        # Chạy forward pass dưới dạng float16
        with autocast():
            predictions = model(data)
            loss = criterion(predictions, targets)

        optimizer.zero_grad()
        
        # Scale loss và backward (để tránh underflow số quá nhỏ)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # ------------------------------------

        total_loss += loss.item()
        _, predicted = predictions.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        loop.set_description(f"Train Acc: {100.*correct/total:.2f}%")

    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # Validate cũng dùng autocast cho nhanh
            with autocast():
                predictions = model(data)
                loss = criterion(predictions, targets)

            total_loss += loss.item()
            _, predicted = predictions.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total

# --- 4. MAIN ---
def main():
    print(f"🔥 TURBO MODE ACTIVATED")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Workers: {NUM_WORKERS}")
    print(f"   Mixed Precision: ENABLED")
    
    train_ds = ANCDataset(DATASET_ROOT, 'train')
    val_ds = ANCDataset(DATASET_ROOT, 'val')
    
    if len(train_ds) == 0:
        print("❌ Lỗi: Không tìm thấy dữ liệu train.")
        return

    # DataLoader tối ưu
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,         # Đẩy nhanh tốc độ RAM -> VRAM
        persistent_workers=True, # Giữ worker sống giữa các epoch (Rất quan trọng trên Windows)
        prefetch_factor=2        # Load trước 2 batch để GPU không phải chờ
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"   Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scaler cho Mixed Precision
    scaler = GradScaler()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("   💾 Model Saved!")

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig('training_result.png')
    print("\n✅ DONE.")

if __name__ == "__main__":
    main()
# Voice Processing Project: ANC & ENC  
Đây là đồ án phục vụ môn học xử lý tiếng nói  
Chủ đề chính: xử lý tín hiệu âm thanh tập trung vào hai bài toán chính: Active Noise Cancellation (ANC) và Environmental Noise Cancellation (ENC) sử dụng thuật toán kết hợp deep learning.  

# Nhóm tác giả  
23110219-Võ Trí Hiệu		      
23110221-Phạm Thiên Hoàng	  
23110239-Nguyễn Quốc Khánh	  
23110315-Lê Ngô Nhựt Tân		  


# Giới thiệu
Dự án bao gồm:

- ANC (Active Noise Cancellation): Mô phỏng quá trình khử ồn chủ động sử dụng thuật toán FxNLMS kết hợp với các mô hình Deep Learning (CNN, ResNet) để chọn bộ lọc tối ưu (Pretrained Control Filters) dựa trên đặc trưng tiếng ồn đầu vào.

- ENC (Environmental Noise Cancellation): Khử nhiễu môi trường (như tiếng xe, tiếng quạt, tiếng TV...) khỏi tín hiệu giọng nói sử dụng mô hình Denoising AutoEncoder.

## Dataset tham khảo: Dữ liệu tiếng ồn và xung phòng được tham khảo từ bộ dữ liệu: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/ETJWLU

Cấu trúc thư mục

```Plaintext
VoiceProcessing_ANC_ENC/
├── data/                           # Chứa dữ liệu mô hình và các file .mat
│   ├── anc/                        # File .mat (Primary/Secondary path, Filters) và model .pth
│   └── enc/                        # Các model .h5 cho ENC (Household, Vehicles,...)
├── input_audio/                    # Thư mục chứa file âm thanh đầu vào (.wav)
├── src/                            # Mã nguồn chính
│   ├── anc/                        # Các module xử lý ANC
│   │   ├── deep_learning/          # Code train model (train.py, prepare_dataset.py)
│   │   ├── fxnlms.py               # Thuật toán FxNLMS
│   │   ├── network.py              # Kiến trúc mạng CNN/ResNet
│   │   └── simulation.py           # Mô phỏng tạo tiếng ồn
│   └── enc/                        # Các module xử lý ENC (config, data_tools)
├── temp_split/                     # Thư mục tạm để xử lý cắt file âm thanh/ảnh spectrogram
├── main_app.py                     # Ứng dụng chính (GUI) chạy mô phỏng
├── view_anc_data_app.py            # Tool xem và kiểm tra dữ liệu bộ lọc 
├── requirements.txt                # Danh sách thư viện cần thiết
└── readme.md                       # Tài liệu hướng dẫn
```
Nhóm sử dụng python 3.13.9 với các thư viện được cài đặt bên dưới

```Bash
pip install -r requirements.txt
```
Lưu ý: Đối với torch và torchaudio, hãy đảm bảo cài đặt phiên bản tương thích với CUDA của máy để tận dụng GPU Nvidia.

# Hướng dẫn sử dụng
## 1. Kiểm tra và xem dữ liệu
Sử dụng tool này để xem biểu đồ các bộ lọc và đáp ứng xung của đường truyền âm thanh từ các file .mat.

Chạy lệnh:
```bash
python view_anc_data_app.py
```
Chức năng:

Tab Control Filters: Load và hiển thị các bộ lọc đã được huấn luyện trước.

Tab Acoustic Paths: Xem đáp ứng xung của đường dẫn sơ cấp (Primary Path) và thứ cấp (Secondary Path).

Hỗ trợ xuất dữ liệu ra file .csv để phân tích thêm.

## 2. Huấn luyện mô hình ANC 
File train nằm trong thư mục src/anc/deep_learning/.

Bước 1: Cấu hình đường dẫn dữ liệu. Mở file src/anc/deep_learning/train.py, tìm và sửa biến DATASET_ROOT trỏ đến thư mục chứa dataset của bạn:


```Python
# Ví dụ trong train.py
DATASET_ROOT = r'Đường_dẫn_tới_thư_mục_Dataset_của_bạn'
```
Bước 2: Chạy quá trình huấn luyện:

```Bash
python src/anc/deep_learning/train.py
```
Mô hình sẽ tự động lưu file best_model.pth khi đạt độ chính xác tốt nhất trên tập Validation.

Biểu đồ Loss/Accuracy sẽ được lưu lại thành training_result.png sau khi train xong.

## 3. Chạy ứng dụng chính 
Đây là giao diện chính tích hợp cả chức năng ANC và ENC.

```Bash
python main_app.py
```
ANC Mode: Chọn file nhiễu, chạy mô phỏng so sánh giữa FxNLMS truyền thống và Hybrid ANC (CNN/ResNet).  
ENC Mode: Chọn loại tiếng ồn (Xe cộ, Gia dụng...) và thực hiện lọc nhiễu cho file ghi âm.

# Một số tài liệu tham khảo  
[1] R. Raphael, “The Filtered-X LMS Algorithm,” Geocities.ws. URL: https://www.geocities.ws/ranjit_raphael/FXLMS.html.  
[2] Z. Luo, D. Shi, và W. S. Gan, “SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning,” GitHub. URL: https://github.com/Luo-Zhengding/SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning.  
[3] Alexander3636, “ENC Speech Enhancement DDAE Base,” Hugging Face. URL:https://huggingface.co/Alexander3636/ENC_Speech_Enhancement_DDAE_Base. 



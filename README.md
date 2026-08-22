# RSNA-Knee-Abnormality-Detection

Pipeline nghiên cứu cho cuộc thi Kaggle **RSNA Knee Abnormality Detection AI Challenge (2026)**.
Cấu trúc thư mục mô phỏng theo repo giảng dạy `dntai_chonnam_ai_theory` (flat, notebook đánh số thứ tự pipeline + 2 file thư viện dùng chung).

## Bối cảnh cuộc thi (FACT — đã xác minh qua web search, nguồn: kaggle.com/competitions/rsna-knee-abnormality-detection, rsna.org)

- Bài toán: 12 finding nhị phân / study, đánh giá bằng macro ROC-AUC.
- Code competition: chạy trong Kaggle Notebook, giới hạn ≤ 9h, không có internet khi chạy submission.
- Dữ liệu: > 5.000 knee MRI exam, ~16–19 site trên thế giới, báo cáo X-quang bằng 12 ngôn ngữ khác nhau.
- **Chất lượng nhãn (quan trọng cho phần indexing/leakage check):** theo một baseline cộng đồng công khai trên GitHub (nguồn thứ cấp, cần tự xác minh lại khi có `train.csv` thật), chỉ khoảng 58/4.407 training study có gold label từ radiologist; phần lớn còn lại chỉ có report-derived weak label, độ khớp với gold label ước tính ~82% — nghĩa là **không nên coi report-derived label có độ tin cậy ngang gold label** khi index hoặc split dữ liệu.

## Cấu trúc hiện tại

```text
.
├── 01. rsna_baseline_v1.ipynb  # khám phá dữ liệu và baseline ban đầu
├── index-data.ipynb            # lập chỉ mục metadata / ảnh
├── RSNA_data.py                # lớp đọc metadata và index dữ liệu
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── train_series.csv
│   ├── test.csv
│   ├── test_series.csv
│   └── sample_submission.csv
└── *.png                       # ảnh kết quả minh họa từ notebook
```

Dữ liệu DICOM và các file cache lớn không nên commit vào Git. Quy tắc loại trừ
được khai báo trong `.gitignore`.

## Cài đặt

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install opencv-python  # RSNA_data.py dùng module cv2
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install opencv-python
```

Mở workspace bằng VS Code, chọn interpreter `.venv`, sau đó chạy các cell trong
`index-data.ipynb` hoặc `01. rsna_baseline_v1.ipynb` theo thứ tự.

## Dữ liệu

Các CSV mẫu đã có trong `data/`. Với dữ liệu đầy đủ từ Kaggle, tải dữ liệu sau
khi đã chấp nhận điều khoản cuộc thi và cấu hình Kaggle API:

```bash
kaggle competitions download -c rsna-knee-abnormality-detection
```

`RSNA_DATASET` mặc định tìm `train.csv` và `train_series.csv` trong `root_path`.
Khi CSV nằm trong thư mục `data/`, truyền `root_path="data"` hoặc truyền đường
dẫn CSV tương ứng. Thư mục ảnh tùy chọn có dạng:

```text
train_images/<StudyInstanceUID>/<SeriesInstanceUID>/<image files>
```

## Quy trình

1. Đọc và kiểm tra metadata trong `data/`.
2. Chạy `index-data.ipynb` để tạo index study/series và, nếu cần, index ảnh.

Khi chia dữ liệu, dùng split ở cấp study/patient để hạn chế leakage. Các nhãn
thiếu được giữ dưới dạng `NaN`; cần kiểm tra chất lượng nhãn trước khi huấn luyện.

## Tham khảo

- [Kaggle: RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [RSNA Knee MRI AI Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)

# RSNA-Knee-Abnormality-Detection

Pipeline nghiên cứu cho cuộc thi Kaggle **RSNA Knee Abnormality Detection AI Challenge (2026)**.
Cấu trúc thư mục mô phỏng theo repo giảng dạy `dntai_chonnam_ai_theory` (flat, notebook đánh số thứ tự pipeline + 2 file thư viện dùng chung).

## Bối cảnh cuộc thi (FACT — đã xác minh qua web search, nguồn: kaggle.com/competitions/rsna-knee-abnormality-detection, rsna.org)

- Bài toán: 12 finding nhị phân / study, đánh giá bằng macro ROC-AUC.
- Code competition: chạy trong Kaggle Notebook, giới hạn ≤ 9h, không có internet khi chạy submission.
- Dữ liệu: > 5.000 knee MRI exam, ~16–19 site trên thế giới, báo cáo X-quang bằng 12 ngôn ngữ khác nhau.
- **Chất lượng nhãn (quan trọng cho phần indexing/leakage check):** theo một baseline cộng đồng công khai trên GitHub (nguồn thứ cấp, cần tự xác minh lại khi có `train.csv` thật), chỉ khoảng 58/4.407 training study có gold label từ radiologist; phần lớn còn lại chỉ có report-derived weak label, độ khớp với gold label ước tính ~82% — nghĩa là **không nên coi report-derived label có độ tin cậy ngang gold label** khi index hoặc split dữ liệu.

## Cấu trúc thư mục

```
RSNA-Knee-Abnormality-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── data/                          # ghi chú nguồn dữ liệu, KHÔNG chứa file DICOM gốc (quá lớn)
│   └── README.md
├── dataset/
│   └── rsna_knee/                 # nơi mount/tải dữ liệu thật về local (đã .gitignore, chỉ giữ .gitkeep)
├── images/                        # hình minh hoạ dùng trong notebook/README
├── rsna_data.py                   # hàm load dữ liệu (metadata CSV, DICOM path, report)
├── rsna_lib.py                    # hàm tiện ích dùng chung (kiểm tra duplicate, patient-level split, tóm tắt chất lượng nhãn)
├── 01. rsna_eda_metadata.ipynb           # khảo sát cấu trúc thư mục & metadata thật (PHASE 0 — không đoán schema)
├── 02. rsna_dicom_indexing.ipynb         # xây file index (manifest) từ DICOM + metadata
├── 03. rsna_label_quality.ipynb          # phân tích gold vs weak label, kiểm tra leakage patient/study-level
└── 04. rsna_baseline_model.ipynb         # baseline model (đơn giản trước, theo nguyên tắc baseline-first)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Tải dữ liệu (cần đã accept rules cuộc thi + cấu hình ~/.kaggle/kaggle.json)
kaggle competitions download -c rsna-knee-abnormality-detection -p dataset/rsna_knee
unzip dataset/rsna_knee/*.zip -d dataset/rsna_knee
```

## Ghi chú reproducibility

- Random seed cố định trong `rsna_lib.py` (mặc định 42, có thể override).
- Mọi số liệu về schema thật (tên cột, cấu trúc thư mục DICOM) PHẢI được xác nhận trong `01. rsna_eda_metadata.ipynb` trước khi dùng ở các notebook sau — không hard-code giả định.
- File index sinh ra ở bước 02 nên được coi là DATA-001 trong traceability, ghi rõ ngày tạo + phiên bản dataset.

## Nguồn tham khảo

- Kaggle competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- RSNA thông báo chính thức: https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge

"""
rsna_data.py
------------
Hàm load dữ liệu cho pipeline RSNA Knee Abnormality Detection.

QUAN TRỌNG: schema thật (tên cột CSV, cấu trúc thư mục DICOM) CHƯA được xác nhận
tại thời điểm viết file này — vì chưa có quyền truy cập dataset thật. Mọi hàm bên
dưới dùng logic MẶC ĐỊNH thường gặp ở các Kaggle DICOM competition, không phải suy
diễn từ trang "Data" chính thức của rsna-knee-abnormality-detection. PHẢI chạy
`01. rsna_eda_metadata.ipynb` để xác nhận trước khi tin tưởng các hàm này.
"""

import os
import glob
import pandas as pd


def get_dataset_root(env_var: str = "RSNA_KNEE_ROOT", default: str = "dataset/rsna_knee") -> str:
    """Trả về đường dẫn gốc dataset. Ưu tiên biến môi trường, fallback về default.

    Raises:
        FileNotFoundError: nếu thư mục không tồn tại — không âm thầm tạo thư mục rỗng
        rồi để pipeline chạy "thành công" trên dữ liệu trống.
    """
    root = os.environ.get(env_var, default)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Không tìm thấy dataset tại '{root}'. "
            f"Tải dữ liệu trước bằng: kaggle competitions download -c rsna-knee-abnormality-detection -p {root}"
        )
    return root


def list_metadata_files(root: str) -> list:
    """Liệt kê toàn bộ file CSV/JSON metadata trong dataset — dùng ở PHASE 0 (inspect trước khi đoán schema)."""
    patterns = ["*.csv", "*.json", "**/*.csv", "**/*.json"]
    found = []
    for p in patterns:
        found.extend(glob.glob(os.path.join(root, p), recursive=True))
    return sorted(set(found))


def load_csv_safe(path: str, n_preview: int = 5) -> pd.DataFrame:
    """Đọc CSV và in preview cột thật — dùng để xác nhận schema thay vì giả định tên cột."""
    df = pd.read_csv(path)
    print(f"[{path}] shape={df.shape}")
    print(f"Cột thật: {list(df.columns)}")
    print(df.head(n_preview))
    return df


def list_dicom_files(root: str) -> list:
    """Liệt kê toàn bộ file .dcm trong dataset (đệ quy). Có thể chậm với dataset lớn —
    cân nhắc dùng generator/batch nếu số lượng file > vài trăm nghìn."""
    return glob.glob(os.path.join(root, "**", "*.dcm"), recursive=True)

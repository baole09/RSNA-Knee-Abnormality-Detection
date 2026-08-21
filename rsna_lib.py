"""
rsna_lib.py
-----------
Hàm tiện ích dùng chung: kiểm tra duplicate/leakage, patient-level split,
tóm tắt chất lượng nhãn (gold vs weak label).

Nguyên tắc: KHÔNG mặc định random image-level/study-level split là an toàn.
Nếu thiếu patient_id, phải báo rõ giới hạn thay vì im lặng bỏ qua leakage check.
"""

import pandas as pd
import numpy as np

RANDOM_SEED = 42


def check_duplicate_ids(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Trả về các dòng có id trùng lặp theo id_col (vd: study_id, patient_id)."""
    if id_col not in df.columns:
        raise KeyError(f"Cột '{id_col}' không tồn tại trong DataFrame. Cột thật: {list(df.columns)}")
    dup_mask = df[id_col].duplicated(keep=False)
    return df[dup_mask].sort_values(id_col)


def patient_level_split(df: pd.DataFrame, patient_col: str, test_size: float = 0.2,
                         seed: int = RANDOM_SEED):
    """Chia train/val theo patient-level (group-aware), tránh 1 patient xuất hiện ở cả 2 tập.

    Nếu không có patient_col trong dữ liệu (nhiều competition y tế ẩn patient_id để
    chống re-identification), hàm sẽ raise lỗi rõ ràng thay vì fallback âm thầm về
    random split — vì random image/study-level split có thể gây leakage nếu 1 patient
    có nhiều study.
    """
    if patient_col not in df.columns:
        raise KeyError(
            f"Không tìm thấy '{patient_col}'. Nếu dataset không cung cấp patient_id, "
            f"CHỈ có thể đảm bảo split an toàn ở study-level — cần ghi rõ giới hạn này "
            f"trong báo cáo, không được mặc định coi là an toàn ở patient-level."
        )
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df[patient_col]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def summarize_label_quality(df: pd.DataFrame, label_source_col: str) -> pd.DataFrame:
    """Đếm số study theo nguồn nhãn (gold / report_weak / none).

    Bối cảnh (cần xác nhận lại với train.csv thật): theo nguồn cộng đồng công khai,
    chỉ ~58/4407 training study có gold label; phần lớn còn lại là report-derived
    weak label với độ khớp ước tính ~82% so với gold. KHÔNG coi 2 loại nhãn này
    có độ tin cậy ngang nhau khi train/evaluate.
    """
    if label_source_col not in df.columns:
        raise KeyError(f"Cột '{label_source_col}' không tồn tại. Cột thật: {list(df.columns)}")
    counts = df[label_source_col].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})


def check_missing_files(df: pd.DataFrame, path_col: str) -> pd.DataFrame:
    """Trả về các dòng có đường dẫn file không tồn tại trên đĩa."""
    import os
    if path_col not in df.columns:
        raise KeyError(f"Cột '{path_col}' không tồn tại. Cột thật: {list(df.columns)}")
    missing_mask = ~df[path_col].apply(lambda p: os.path.isfile(str(p)))
    return df[missing_mask]

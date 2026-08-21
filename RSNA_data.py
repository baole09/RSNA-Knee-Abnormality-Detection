"""
RSNA_data.py

Metadata indexing and lazy image access for the RSNA knee MRI dataset.

Expected files:
    train.csv
    train_series.csv

Optional image layout:
    train_images/
        <StudyInstanceUID>/
            <SeriesInstanceUID>/
                <image files>

Main tables:
    db_study  : one row per StudyInstanceUID
    db_series : one row per SeriesInstanceUID
    db_image  : one row per image (optional)

Design:
    Study -> Series -> Image

Only metadata is loaded into RAM. MRI pixels are loaded on demand.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd


class RSNA_DATASET:
    LABEL_COLUMNS = [
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]

    SERIES_COLUMNS = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    ]

    def __init__(
        self,
        root_path=".",
        train_csv="train.csv",
        train_series_csv="train_series.csv",
        image_root="train_images",
        cache_name="rsna_cache.h5",
        build_image_index=False,
        force_rebuild=False,
        add_split=False,
        test_size=0.2,
        random_state=42,
    ):
        self.root_path = os.path.abspath(root_path)

        self.train_csv = os.path.join(self.root_path, train_csv)
        self.train_series_csv = os.path.join(
            self.root_path, train_series_csv
        )
        self.image_root = os.path.join(
            self.root_path, image_root
        )
        self.cache_file = os.path.join(
            self.root_path, cache_name
        )

        self.db_study = None
        self.db_series = None
        self.db_image = None

        if force_rebuild or not os.path.exists(self.cache_file):
            self.build(
                build_image_index=build_image_index,
                add_split=add_split,
                test_size=test_size,
                random_state=random_state,
            )
        else:
            self.load_cache()

    # ============================================================
    # BUILD
    # ============================================================

    def build(
        self,
        build_image_index=False,
        add_split=False,
        test_size=0.2,
        random_state=42,
    ):
        """Build metadata indexes and save them to HDF5."""

        print("+ Building RSNA dataset index")

        if not os.path.exists(self.train_csv):
            raise FileNotFoundError(
                "Cannot find: " + self.train_csv
            )

        if not os.path.exists(self.train_series_csv):
            raise FileNotFoundError(
                "Cannot find: " + self.train_series_csv
            )

        self.db_study = pd.read_csv(self.train_csv)
        self.db_series = pd.read_csv(self.train_series_csv)

        self._validate_columns()

        # Convert labels to numeric.
        # NaN is intentionally preserved for unlabeled studies.
        for col in self.LABEL_COLUMNS:
            self.db_study[col] = pd.to_numeric(
                self.db_study[col],
                errors="coerce",
            )

        # How many of the 12 labels exist for this study?
        self.db_study["num_labels"] = (
            self.db_study[self.LABEL_COLUMNS]
            .notna()
            .sum(axis=1)
            .astype(np.int16)
        )

        self.db_study["is_labeled"] = (
            self.db_study["num_labels"] > 0
        ).astype(np.int8)

        # Count series belonging to each study.
        series_count = (
            self.db_series
            .groupby("StudyInstanceUID")
            .size()
            .rename("num_series")
        )

        self.db_study = self.db_study.merge(
            series_count,
            how="left",
            left_on="StudyInstanceUID",
            right_index=True,
        )

        self.db_study["num_series"] = (
            self.db_study["num_series"]
            .fillna(0)
            .astype(np.int16)
        )

        if add_split:
            self.add_train_test_split(
                test_size=test_size,
                random_state=random_state,
            )

        print("db_study :", self.db_study.shape)
        print("db_series:", self.db_series.shape)

        if build_image_index:
            self.db_image = self.build_image_index()
            print("db_image :", self.db_image.shape)

        self.save_cache()

    # ============================================================
    # CACHE
    # ============================================================

    def save_cache(self):
        """Save all available tables to one HDF5 file."""

        if self.db_study is not None:
            self.db_study.to_hdf(
                self.cache_file,
                key="db_study",
                format="table",
                mode="w",
            )

        if self.db_series is not None:
            self.db_series.to_hdf(
                self.cache_file,
                key="db_series",
                format="table",
                mode="a",
            )

        if self.db_image is not None:
            self.db_image.to_hdf(
                self.cache_file,
                key="db_image",
                format="table",
                mode="a",
            )

        print("+ Cache saved:", self.cache_file)

    def load_cache(self):
        """Load previously indexed tables."""

        print("+ Loading cache:", self.cache_file)

        with pd.HDFStore(self.cache_file, mode="r") as store:

            keys = set(store.keys())

            if "/db_study" in keys:
                self.db_study = store["db_study"]

            if "/db_series" in keys:
                self.db_series = store["db_series"]

            if "/db_image" in keys:
                self.db_image = store["db_image"]

        if self.db_study is None:
            raise RuntimeError(
                "db_study is missing from cache."
            )

        if self.db_series is None:
            raise RuntimeError(
                "db_series is missing from cache."
            )

        print("db_study :", self.db_study.shape)
        print("db_series:", self.db_series.shape)

        if self.db_image is not None:
            print("db_image :", self.db_image.shape)

    def cache_info(self):
        """Display HDF5 cache information."""

        with pd.HDFStore(
            self.cache_file,
            mode="r",
        ) as store:
            print(store)

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_columns(self):

        required_study = [
            "StudyInstanceUID",
            "Report",
            *self.LABEL_COLUMNS,
        ]

        required_series = self.SERIES_COLUMNS

        missing_study = [
            col
            for col in required_study
            if col not in self.db_study.columns
        ]

        missing_series = [
            col
            for col in required_series
            if col not in self.db_series.columns
        ]

        if missing_study:
            raise ValueError(
                "Missing columns in train.csv: "
                + str(missing_study)
            )

        if missing_series:
            raise ValueError(
                "Missing columns in train_series.csv: "
                + str(missing_series)
            )

    # ============================================================
    # STUDY INDEX
    # ============================================================

    def get_study(self, study_uid):
        """Return one StudyInstanceUID row."""

        result = self.db_study[
            self.db_study["StudyInstanceUID"] == study_uid
        ]

        if len(result) == 0:
            return None

        return result.iloc[0]

    def get_study_ids(self):
        """Return all StudyInstanceUID values."""

        return self.db_study[
            "StudyInstanceUID"
        ].to_numpy()

    # ============================================================
    # SERIES INDEX
    # ============================================================

    def get_series(self, study_uid):
        """Return every series belonging to one study."""

        return (
            self.db_series[
                self.db_series["StudyInstanceUID"] == study_uid
            ]
            .reset_index(drop=True)
        )

    def get_series_by_uid(self, series_uid):
        """Return one SeriesInstanceUID row."""

        result = self.db_series[
            self.db_series["SeriesInstanceUID"] == series_uid
        ]

        if len(result) == 0:
            return None

        return result.iloc[0]

    def get_study_bundle(self, study_uid):
        """
        Return:
            study metadata
            series metadata
            labels

        No MRI pixels are loaded.
        """

        study = self.get_study(study_uid)

        if study is None:
            return None

        return {
            "StudyInstanceUID": study_uid,
            "study": study,
            "series": self.get_series(study_uid),
            "labels": self.get_labels(study_uid),
        }

    # ============================================================
    # LABELS
    # ============================================================

    def get_labels(self, study_uid):
        """Return the 12 diagnosis labels."""

        study = self.get_study(study_uid)

        if study is None:
            return None

        return study[
            self.LABEL_COLUMNS
        ].to_numpy(dtype=np.float32)

    def get_label_dataframe(self):
        """Return StudyInstanceUID + 12 labels."""

        return self.db_study[
            ["StudyInstanceUID"] + self.LABEL_COLUMNS
        ].copy()

    # ============================================================
    # TRAIN / TEST SPLIT
    # ============================================================

    def add_train_test_split(
        self,
        test_size=0.2,
        random_state=42,
    ):
        """
        Split at STUDY level.

        0 = train
        1 = test

        Never split individual MRI images/series independently.
        """

        if not 0 < test_size < 1:
            raise ValueError(
                "test_size must be between 0 and 1."
            )

        rng = np.random.default_rng(
            random_state
        )

        n = len(self.db_study)

        indices = np.arange(n)
        rng.shuffle(indices)

        n_test = int(round(n * test_size))

        test_indices = indices[:n_test]

        split = np.zeros(
            n,
            dtype=np.int8,
        )

        split[test_indices] = 1

        self.db_study[
            "train_test"
        ] = split

    # ============================================================
    # SERIES FILTERING
    # ============================================================

    def filter_series(
        self,
        anatomical_plane=None,
        fluid_sensitive=None,
        fat_suppression=None,
    ):
        """
        Filter MRI series.

        Example:

            dataset.filter_series(
                anatomical_plane="Sagittal",
                fluid_sensitive=1,
                fat_suppression=1
            )
        """

        df = self.db_series

        if anatomical_plane is not None:

            if isinstance(
                anatomical_plane,
                str,
            ):
                anatomical_plane = [
                    anatomical_plane
                ]

            df = df[
                df["Anatomical_Plane"].isin(
                    anatomical_plane
                )
            ]

        if fluid_sensitive is not None:
            df = df[
                df["Fluid_Sensitive"]
                == fluid_sensitive
            ]

        if fat_suppression is not None:
            df = df[
                df["Fat_Suppression"]
                == fat_suppression
            ]

        return df.reset_index(drop=True)

    def make_sample_index(
        self,
        labeled_only=False,
        anatomical_plane=None,
        fluid_sensitive=None,
        fat_suppression=None,
    ):
        """
        Create the main ML indexing table.

        One row = one StudyInstanceUID.

        This is the table that should normally be passed to
        a PyTorch Dataset.
        """

        df = self.db_study.copy()

        if labeled_only:
            df = df[
                df["is_labeled"] == 1
            ]

        if (
            anatomical_plane is not None
            or fluid_sensitive is not None
            or fat_suppression is not None
        ):

            series = self.filter_series(
                anatomical_plane=anatomical_plane,
                fluid_sensitive=fluid_sensitive,
                fat_suppression=fat_suppression,
            )

            valid_studies = (
                series["StudyInstanceUID"]
                .unique()
            )

            df = df[
                df["StudyInstanceUID"].isin(
                    valid_studies
                )
            ]

        columns = [
            "StudyInstanceUID",
            *self.LABEL_COLUMNS,
            "num_labels",
            "is_labeled",
        ]

        if "train_test" in df.columns:
            columns.append("train_test")

        return (
            df[columns]
            .reset_index(drop=True)
        )

    # ============================================================
    # IMAGE INDEX
    # ============================================================

    def build_image_index(self):
        """
        Scan:

            train_images/
                StudyInstanceUID/
                    SeriesInstanceUID/
                        images...

        Creates one metadata row per image.

        IMPORTANT:
            image pixels are NOT loaded.
        """

        if not os.path.isdir(
            self.image_root
        ):
            raise FileNotFoundError(
                "Image directory does not exist: "
                + self.image_root
            )

        rows = []

        # Use train_series.csv as the source
        # instead of blindly scanning everything.
        for row in self.db_series.itertuples(
            index=False
        ):

            study_uid = row.StudyInstanceUID
            series_uid = row.SeriesInstanceUID

            series_dir = os.path.join(
                self.image_root,
                str(study_uid),
                str(series_uid),
            )

            if not os.path.isdir(
                series_dir
            ):
                continue

            image_files = [
                p
                for p in glob.glob(
                    os.path.join(
                        series_dir,
                        "*",
                    )
                )
                if os.path.isfile(p)
            ]

            image_files.sort()

            for frame_idx, image_path in enumerate(
                image_files
            ):

                rows.append(
                    {
                        "StudyInstanceUID": study_uid,
                        "SeriesInstanceUID": series_uid,
                        "frame_idx": frame_idx,
                        "image_name": os.path.basename(
                            image_path
                        ),
                        "image_path": os.path.relpath(
                            image_path,
                            self.root_path,
                        ),
                    }
                )

        return pd.DataFrame(
            rows,
            columns=[
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "frame_idx",
                "image_name",
                "image_path",
            ],
        )

    def get_image_rows(self, series_uid):
        """Return all indexed images of a series."""

        if self.db_image is None:
            raise RuntimeError(
                "db_image does not exist. "
                "Initialize with "
                "build_image_index=True."
            )

        return (
            self.db_image[
                self.db_image["SeriesInstanceUID"]
                == series_uid
            ]
            .sort_values("frame_idx")
            .reset_index(drop=True)
        )

    def get_image_path(
        self,
        series_uid,
        frame_idx,
    ):
        """Return the absolute path of one MRI image."""

        rows = self.get_image_rows(
            series_uid
        )

        if (
            frame_idx < 0
            or frame_idx >= len(rows)
        ):
            raise IndexError(
                f"frame_idx={frame_idx} is invalid. "
                f"Number of images={len(rows)}"
            )

        relative_path = rows.iloc[
            frame_idx
        ]["image_path"]

        return os.path.join(
            self.root_path,
            relative_path,
        )

    def load_image(
        self,
        series_uid,
        frame_idx=0,
        flags=cv2.IMREAD_ANYDEPTH,
    ):
        """
        Load ONE MRI image.

        This is the memory-efficient approach:
        do not load all MRI images into RAM.
        """

        image_path = self.get_image_path(
            series_uid,
            frame_idx,
        )

        image = cv2.imread(
            image_path,
            flags,
        )

        if image is None:
            raise IOError(
                "Cannot read image: "
                + image_path
            )

        return image

    # ============================================================
    # EDA / SUMMARY
    # ============================================================

    def summary(self):
        """Print useful dataset statistics."""

        print(
            "\n========== RSNA DATASET SUMMARY =========="
        )

        print(
            f"Studies : {len(self.db_study):,}"
        )

        print(
            f"Series  : {len(self.db_series):,}"
        )

        if self.db_image is not None:
            print(
                f"Images  : {len(self.db_image):,}"
            )

        print(
            "\n--- Label availability ---"
        )

        print(
            self.db_study[
                self.LABEL_COLUMNS
            ]
            .notna()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        print(
            "\n--- Positive labels ---"
        )

        print(
            self.db_study[
                self.LABEL_COLUMNS
            ]
            .sum(skipna=True)
            .sort_values(
                ascending=False
            )
        )

        print(
            "\n--- Anatomical plane ---"
        )

        print(
            self.db_series[
                "Anatomical_Plane"
            ]
            .value_counts(
                dropna=False
            )
        )

        print(
            "\n--- Fluid Sensitive ---"
        )

        print(
            self.db_series[
                "Fluid_Sensitive"
            ]
            .value_counts(
                dropna=False
            )
        )

        print(
            "\n--- Fat Suppression ---"
        )

        print(
            self.db_series[
                "Fat_Suppression"
            ]
            .value_counts(
                dropna=False
            )
        )

        print(
            "===========================================\n"
        )

    # ============================================================
    # PYTORCH-FRIENDLY ACCESS
    # ============================================================

    def __len__(self):
        return len(self.db_study)

    def __getitem__(self, idx):
        """
        Metadata-only sample access.

        No MRI image is loaded.
        """

        row = self.db_study.iloc[idx]

        study_uid = (
            row["StudyInstanceUID"]
        )

        return {
            "StudyInstanceUID": study_uid,
            "labels": row[
                self.LABEL_COLUMNS
            ].to_numpy(
                dtype=np.float32
            ),
            "series": self.get_series(
                study_uid
            ),
        }


if __name__ == "__main__":

    # Example for Kaggle:
    #
    # dataset = RSNA_DATASET(
    #     root_path="/kaggle/input/your-dataset",
    #     build_image_index=False,
    # )
    #
    # dataset.summary()
    #
    # sample = dataset[0]
    #
    # print(
    #     sample["StudyInstanceUID"]
    # )
    #
    # print(sample["labels"])
    #
    # print(sample["series"])

    print(
        "RSNA_DATASET loaded successfully."
    )

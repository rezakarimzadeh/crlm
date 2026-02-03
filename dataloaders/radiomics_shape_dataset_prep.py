
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, matched_df, feature_to_include, dataloader_config):
        '''

        '''
        self.matched_df = matched_df
        self.feature_to_include = feature_to_include
        self.dataloader_config = dataloader_config
        self.aggregated_data = self.prepare_data()

    def __len__(self):
        return len(self.aggregated_data)

    def read_pre_post_treatment_csv(self, patient_id, features_base_dir, postfix):
        base_csv_path = features_base_dir / f"{patient_id}_0_0000_{postfix}.csv"
        followup_csv_path = features_base_dir / f"{patient_id}_1_0000_{postfix}.csv"
        base_features = pd.read_csv(base_csv_path)
        followup_features = pd.read_csv(followup_csv_path)
        return base_features, followup_features
    
    def filter_features_based_on_inclusion(self, features_df):
        features_to_add = []
        for feature_type in self.feature_to_include:
            if feature_type not in self.dataloader_config:
                raise ValueError(f"Feature type {feature_type} not found in dataloader config.")
            features_to_add.extend(self.dataloader_config[feature_type])
        filtered_df = features_df[features_to_add]
        lesion_labels = features_df['lesion_label']
        return filtered_df, lesion_labels
    
    
    def aggregate_radiomics_and_shape_features(self, patient_id):
        radiomics_features_dir = Path(self.dataloader_config['radiomics_features_dir'])
        shape_features_dir = Path(self.dataloader_config['extra_shape_features_dir'])
        radiomics_base, radiomics_followup = self.read_pre_post_treatment_csv(patient_id, radiomics_features_dir, 'radiomics')
        shape_base, shape_followup = self.read_pre_post_treatment_csv(patient_id, shape_features_dir, 'extra_shape_features')
        
        base_img_features_df = pd.merge(radiomics_base, shape_base, on='lesion_label')
        followup_img_features_df = pd.merge(radiomics_followup, shape_followup, on='lesion_label')

        base_img_features_df, base_lesion_labels = self.filter_features_based_on_inclusion(base_img_features_df)
        followup_img_features_df, followup_lesion_labels = self.filter_features_based_on_inclusion(followup_img_features_df)

        output = {
            'base_img_features': base_img_features_df.values,
            'base_lesion_labels': base_lesion_labels.values,
            'followup_img_features': followup_img_features_df.values,
            'followup_lesion_labels': followup_lesion_labels.values
        }
        return output
    
    def prepare_data(self):
        mut_map = {
                        'BRAF mutation': 0,
                        'RAS & BRAF wildtype': 1,
                        'RAS mutation': 2
                    } 
        
        sex_map = { 'Female': 0, 'Male': 1 }
        who_map = { 0:0, 1:1, 2:2}
        early_response_map = { 0:0, 1:1}
        
        aggregated_data = []
        print("Aggregating radiomics and shape features for all patients...")
        for idx, row in tqdm(self.matched_df.iterrows(), total=len(self.matched_df)):
            patient_id = row['patient_id']
            try:
                output = self.aggregate_radiomics_and_shape_features(patient_id)
                # print(f"Aggregated features for patient {patient_id}: {output['base_img_features'].shape[0]} lesions in base, {output['followup_img_features'].shape[0]} lesions in follow-up.")
            except Exception as e:
                print(f"Error processing patient {patient_id}: {e}")
                continue
            overall_survival = int(row['OSm'] > 24)  # example threshold
              
            demographic_info = [sex_map.get(row['sex'], -1), row['Age'], who_map.get(row['WHO'], -1),  mut_map.get(row['mutstat'], -1)]
            aggregated_data.append({ "patient_id": patient_id,
                                     "base_img_features": output['base_img_features'],
                                     "base_lesion_labels": output['base_lesion_labels'],
                                     "followup_img_features": output['followup_img_features'],
                                     "followup_lesion_labels": output['followup_lesion_labels'],
                                     "early_response": early_response_map.get(row['ER (1 = yes, 0 = no)'], 0),
                                     "overall_survival_24m": overall_survival,
                                     "demographic_info": demographic_info,
                                     "all clinical_info": row.to_dict()
                                      })
            print(f"Processed patient {patient_id}, early_response: {row['ER (1 = yes, 0 = no)']}, overall_survival_24m: {overall_survival}")
        return aggregated_data  

    def __getitem__(self, idx):
        data_dict = self.aggregated_data[idx]
        return data_dict
    

import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import joblib


def _make_cache_key(matched_df: pd.DataFrame, feature_to_include, dataloader_config: dict) -> str:
    """
    Create a stable cache key for the prepared dataset, based on:
      - ordered patient ids
      - requested feature groups
      - dataloader config (paths + feature lists)
    """
    payload = {
        "patient_ids": matched_df["patient_id"].astype(str).tolist(),
        "feature_to_include": list(feature_to_include),
        "dataloader_config": dataloader_config,
    }
    s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(s).hexdigest()


class FastCustomDataset(Dataset):
    def __init__(
        self,
        matched_df: pd.DataFrame,
        feature_to_include,
        dataloader_config: dict,
        cache_dir: str = './cached_data',
        use_cache: bool = False,
        verbose: bool = False,
    ):
        """
        matched_df: dataframe with columns like:
          patient_id, sex, Age, WHO, mutstat, OSm, ER (1 = yes, 0 = no), ...
        feature_to_include: list of feature group keys that exist in dataloader_config
        dataloader_config: dict containing:
          - 'radiomics_features_dir'
          - 'extra_shape_features_dir'
          - feature group lists (e.g. 'shape', 'firstorder', etc.)
          - optional: 'prep_num_workers'
        cache_dir: directory to store prepared aggregated_data
        """
        self.matched_df = matched_df.reset_index(drop=True).copy()
        self.feature_to_include = list(feature_to_include)
        self.dataloader_config = dict(dataloader_config)
        self.verbose = verbose

        # ---- Precompute feature list once ----
        self.features_to_add = self._compute_features_to_add()

        # Optional: allow tuning
        self.prep_num_workers = int(self.dataloader_config.get("prep_num_workers", 8))

        # ---- Cache handling ----
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = bool(use_cache and self.cache_dir is not None)

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            key = _make_cache_key(self.matched_df, self.feature_to_include, self.dataloader_config)
            self.cache_path = self.cache_dir / f"aggregated_{key}.joblib"
            if self.cache_path.exists():
                self.aggregated_data = joblib.load(self.cache_path)
                if self.verbose:
                    print(f"[Cache] Loaded aggregated_data from: {self.cache_path}")
                return

        # ---- Prepare data (expensive) ----
        self.aggregated_data = self.prepare_data()

        if self.use_cache:
            joblib.dump(self.aggregated_data, self.cache_path, compress=3)
            if self.verbose:
                print(f"[Cache] Saved aggregated_data to: {self.cache_path}")

    def __len__(self):
        return len(self.aggregated_data)

    def __getitem__(self, idx):
        return self.aggregated_data[idx]

    # ----------------------------
    # Feature selection utilities
    # ----------------------------
    def _compute_features_to_add(self):
        features_to_add = []
        for feature_type in self.feature_to_include:
            if feature_type not in self.dataloader_config:
                raise ValueError(f"Feature type {feature_type} not found in dataloader config.")
            features_to_add.extend(self.dataloader_config[feature_type])
        # de-duplicate while preserving order
        seen = set()
        out = []
        for f in features_to_add:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    def _read_csv_fast(self, path: Path, wanted_cols_set: set) -> pd.DataFrame:
        """
        Read only lesion_label + wanted columns from CSV to minimize IO.
        Works even if some wanted columns aren't present in that CSV.
        """
        return pd.read_csv(
            path,
            usecols=lambda c: (c == "lesion_label") or (c in wanted_cols_set),
            engine="c",
            low_memory=False,
        )

    def read_pre_post_treatment_csv(self, patient_id: str, features_base_dir: Path, postfix: str, wanted_cols_set: set):
        base_csv_path = features_base_dir / f"{patient_id}_0_0000_{postfix}.csv"
        followup_csv_path = features_base_dir / f"{patient_id}_1_0000_{postfix}.csv"
        base_features = self._read_csv_fast(base_csv_path, wanted_cols_set)
        followup_features = self._read_csv_fast(followup_csv_path, wanted_cols_set)
        return base_features, followup_features

    def filter_features_based_on_inclusion(self, features_df: pd.DataFrame):
        # by now features_df should contain lesion_label and selected feature columns
        filtered_df = features_df[self.features_to_add]
        lesion_labels = features_df["lesion_label"]
        return filtered_df, lesion_labels

    # ----------------------------
    # Aggregation per patient
    # ----------------------------
    def aggregate_radiomics_and_shape_features(self, patient_id: str):
        radiomics_features_dir = Path(self.dataloader_config["radiomics_features_dir"])
        shape_features_dir = Path(self.dataloader_config["extra_shape_features_dir"])

        wanted_cols_set = set(self.features_to_add)

        # Read minimal columns
        radiomics_base, radiomics_followup = self.read_pre_post_treatment_csv(
            patient_id, radiomics_features_dir, "radiomics", wanted_cols_set
        )
        shape_base, shape_followup = self.read_pre_post_treatment_csv(
            patient_id, shape_features_dir, "extra_shape_features", wanted_cols_set
        )

        # Faster than merge: index and join on lesion_label
        # Keep lesion_label column for output labels
        rb = radiomics_base.set_index("lesion_label", drop=False)
        sb = shape_base.set_index("lesion_label", drop=False)
        base_df = rb.join(sb.drop(columns=["lesion_label"], errors="ignore"), how="inner")

        rf = radiomics_followup.set_index("lesion_label", drop=False)
        sf = shape_followup.set_index("lesion_label", drop=False)
        follow_df = rf.join(sf.drop(columns=["lesion_label"], errors="ignore"), how="inner")

        base_df, base_labels = self.filter_features_based_on_inclusion(base_df)
        follow_df, follow_labels = self.filter_features_based_on_inclusion(follow_df)

        return {
            "base_img_features": base_df.to_numpy(copy=False),
            "base_lesion_labels": base_labels.to_numpy(copy=False),
            "followup_img_features": follow_df.to_numpy(copy=False),
            "followup_lesion_labels": follow_labels.to_numpy(copy=False),
        }

    # ----------------------------
    # Clinical preprocessing (vectorized)
    # ----------------------------
    def _prepare_clinical_rows(self):
        df = self.matched_df.copy()

        mut_map = {
            "BRAF mutation": 0,
            "RAS & BRAF wildtype": 1,
            "RAS mutation": 2,
        }
        sex_map = {"Female": 0, "Male": 1}

        # Map / coerce
        df["mutstat_enc"] = df["mutstat"].map(mut_map).fillna(-1).astype(int)
        df["sex_enc"] = df["sex"].map(sex_map).fillna(-1).astype(int)
        df["who_enc"] = pd.to_numeric(df["WHO"], errors="coerce").fillna(-1).astype(int)
        df["age_f"] = pd.to_numeric(df["Age"], errors="coerce").fillna(-1.0).astype(float)

        df["early_response"] = pd.to_numeric(df["ER (1 = yes, 0 = no)"], errors="coerce").fillna(0).astype(int)

        osm = pd.to_numeric(df["OSm"], errors="coerce")
        df["overall_survival_24m"] = (osm > 24).fillna(False).astype(int)

        rows = df.to_dict(orient="records")
        out_rows = []
        for r in rows:
            demographic_info = [r["sex_enc"], r["age_f"], r["who_enc"], r["mutstat_enc"]]
            out_rows.append(
                {
                    "patient_id": r["patient_id"],
                    "early_response": r["early_response"],
                    "overall_survival_24m": r["overall_survival_24m"],
                    "demographic_info": demographic_info,
                    "all_clinical_info": r,
                }
            )
        return out_rows

    # ----------------------------
    # Main preparation (parallel)
    # ----------------------------
    def prepare_data(self):
        rows = self._prepare_clinical_rows()

        aggregated_data = []
        if self.verbose:
            print("Aggregating radiomics + shape for all patients...")

        def process_one(r):
            patient_id = r["patient_id"]
            output = self.aggregate_radiomics_and_shape_features(patient_id)

            return {
                "patient_id": patient_id,
                "base_img_features": output["base_img_features"],
                "base_lesion_labels": output["base_lesion_labels"],
                "followup_img_features": output["followup_img_features"],
                "followup_lesion_labels": output["followup_lesion_labels"],
                "early_response": r["early_response"],
                "overall_survival_24m": r["overall_survival_24m"],
                "demographic_info": r["demographic_info"],
                "all clinical_info": r["all_clinical_info"],
            }

        # Threads: good for CSV IO. If you see slowdown on /mnt/l, reduce to 4-8.
        with ThreadPoolExecutor(max_workers=self.prep_num_workers) as ex:
            futures = [ex.submit(process_one, r) for r in rows]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Preparing dataset"):
                try:
                    aggregated_data.append(f.result())
                except Exception as e:
                    # Optionally log some info
                    if self.verbose:
                        print(f"[Skip] {e}")
                    continue

        # keep deterministic order if you want (optional):
        # aggregated_data.sort(key=lambda x: str(x["patient_id"]))
        return aggregated_data

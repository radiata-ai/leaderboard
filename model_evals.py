"""
model_evals.py

A script to evaluate a pretrained Brain2Vec model (PCA or VAE) on 
train/validation/test splits, compute reconstruction errors and 
downstream performance, and optionally push/pull results from S3.

Usage (example):
    python /home/ubuntu/model_evals/model_evals.py \
        --model_repo_path /home/ubuntu/brain2vec \
        --checkpoint_file /home/ubuntu/brain2vec/autoencoder_final.pth \
        --inference_script /home/ubuntu/brain2vec/inference_brain2vec.py \
        --pull_model_evals \
        --push_files_to_s3 \
        --s3_output_path s3://radiata-data/leaderboards/brain2vec

    Use precomputed embeddings to run downstream models:
    python model_evals.py \
        --model_repo_path /home/ubuntu/brain2vec_PCA \
        --checkpoint_file pca_model.joblib \
        --inference_script inference_brain2vec_PCA.py \
        --use_precomputed_embeddings \
        --embeddings_file /home/ubuntu/precomputed_features/embeddings.json \
        --pull_model_evals \
        --push_files_to_s3 \
        --s3_output_path s3://radiata-data/leaderboards/brain2vec \
        --eval_results_dir eval_results

    Create a custom features file:
    from model_evals import generate_hf_compatible_embeddings_from_features
    generate_hf_compatible_embeddings_from_features(
        features_file="my_features.csv",
        output_file="my_custom_features.json",
        id_column="nii_filepath",
        metadata_columns=["study","radiata_id"],
        embedding_columns=["TBV","GMV","WMV"]
    )

    python model_evals.py \
        --model_repo_path /home/ubuntu/brain2vec_PCA \
        --checkpoint_file pca_model.joblib \
        --inference_script inference_brain2vec_PCA.py \
        --use_precomputed_embeddings \
        --embeddings_file my_custom_features.json

Required arguments:
  --model_repo_path      Path to the cloned local model repo directory.
  --checkpoint_file      Name of the checkpoint file inside that directory (e.g. 'autoencoder-ep-4.pth' or 'pca_model.joblib').

Optional arguments:
  --pull_model_evals     Pull down an existing model_evals.json from S3 before appending new results (default: off).
  --push_files_to_s3     Push the updated model_evals.json and results folders to S3 (default: off).
  --s3_output_path       S3 path for reading/writing results (default: 's3://radiata-data/leaderboards/brain2vec').
  --eval_results_dir     Folder name where all evaluation results are stored (NPZ, CSV, etc.). Defaults to 'eval_results'.
  --use_precomputed_embeddings  If set, skip model inference. Load embeddings/features from --embeddings_file instead.
  --embeddings_file Path to load/store embeddings. If None and doing inference, defaults to {eval_results_dir}/embeddings.json.

  
Important:
  To pull from or push to S3, AWS credentials must be configured, 
  typically in ~/.aws/credentials or via environment variables.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
from typing import Union, List, Dict
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import joblib
from generative.losses import PerceptualLoss
from torch.nn import L1Loss
from tqdm import tqdm
import boto3
from datasets import load_dataset, concatenate_datasets


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_brain2vec_model(repo_id: str, checkpoint_filename: str, inference_module_name: str, device: str = "cpu"):
    import os
    import importlib.util
    import sys

    # e.g. if repo_id="/home/ubuntu/brain2vec_PCA" and inference_module_name="inference_brain2vec_PCA"
    # Then the script path is "/home/ubuntu/brain2vec_PCA/inference_brain2vec_PCA.py"
    module_path = os.path.join(repo_id, inference_module_name if inference_module_name.endswith('.py') else f"{inference_module_name}.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Cannot find inference script: {module_path}")

    # 1) Dynamically load that file as a module
    spec = importlib.util.spec_from_file_location("repo_model", module_path)
    repo_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repo_model)

    # 2) Check the checkpoint
    ckpt_path = os.path.join(repo_id, checkpoint_filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # 3) Decide if it's PCA (.joblib) or PyTorch (.pth)
    if checkpoint_filename.endswith(".joblib"):
        print(f"[load_brain2vec_model] Detected PCA model => {ckpt_path}")
        model = repo_model.PCABrain2vec.from_pretrained(ckpt_path)
        model.eval()  # no-op for PCA, but safe
    else:
        print(f"[load_brain2vec_model] Detected PyTorch AE => {ckpt_path}")
        model = repo_model.Brain2vec.from_pretrained(checkpoint_path=ckpt_path, device=device)

    # 4) Get preprocess_mri
    preprocess_fn = getattr(repo_model, "preprocess_mri", None)
    if preprocess_fn is None:
        raise AttributeError(f"Could not find 'preprocess_mri' in {module_path}.")

    return model, preprocess_fn


def fit_downstream_tasks(embeddings_train, ages_train, sexes_train, diags_train):
    """
    Fit 3 downstream models on the train embeddings:
      - Age: LinearRegression (only on cognitively_normal)
      - Sex: LDA (only on cognitively_normal)
      - Diagnosis: LDA (on all scans)

    Return the fitted models in a dict.
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    # Convert to numpy
    X = np.array(embeddings_train)
    ages_arr = np.array(ages_train)
    sexes_arr = np.array(sexes_train)
    diags_arr = np.array(diags_train)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1) Subset for cognitively normal
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    normal_mask = (diags_arr == "cognitively_normal")
    #   Age & Sex => only cognitively_normal
    X_norm = X[normal_mask]
    ages_norm = ages_arr[normal_mask]
    sexes_norm = sexes_arr[normal_mask]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2) Fit Age Regression (on normal)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reg_age = LinearRegression()
    if len(X_norm) > 0:
        reg_age.fit(X_norm, ages_norm)
    else:
        # In case no normal samples exist, handle gracefully
        print("Warning: No cognitively_normal samples found for Age Regression.")
        reg_age = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3) Fit Sex Classification (on normal)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from sklearn.preprocessing import LabelEncoder
    le_sex = LabelEncoder()
    if len(X_norm) > 0:
        sexes_norm_encoded = le_sex.fit_transform(sexes_norm)
        lda_sex = LinearDiscriminantAnalysis()
        lda_sex.fit(X_norm, sexes_norm_encoded)
    else:
        print("Warning: No cognitively_normal samples found for Sex Classification.")
        lda_sex = None
        sexes_norm_encoded = None
        # le_sex can be partially or empty

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4) Fit Diagnosis Classification (on ALL)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    le_diag = LabelEncoder()
    diags_encoded = le_diag.fit_transform(diags_arr)
    lda_diag = LinearDiscriminantAnalysis()
    lda_diag.fit(X, diags_encoded)

    # Return all models + label encoders
    models = {
        "reg_age": reg_age,
        "lda_sex": lda_sex,
        "le_sex": le_sex,
        "lda_diag": lda_diag,
        "le_diag": le_diag,
    }
    return models


def compute_per_class_accuracy(y_true, y_pred, label_encoder):
    """
    Given numeric-encoded y_true, y_pred, and a label_encoder,
    compute the fraction of correct predictions for each class.
    Returns a dict {class_label: accuracy_in_that_class}.
    """
    import numpy as np

    classes_in_data = np.unique(y_true)  # e.g. [0,1] if 2 classes
    per_class_acc = {}

    for c in classes_in_data:
        mask = (y_true == c)
        total = np.sum(mask)
        if total == 0:
            # no samples of this class => skip or record None
            continue
        correct = np.sum((y_pred == c) & mask)
        acc = correct / total

        # Map numeric class -> string label
        class_label = label_encoder.inverse_transform([c])[0]
        per_class_acc[class_label] = float(acc)
    
    return per_class_acc


def evaluate_downstream_tasks(models, embeddings, ages, sexes, diags):
    """
    Evaluate the 3 downstream models on a given set (train/val/test).
    
    - Age & Sex => only cognitively_normal samples
    - Diagnosis => all samples
    
    Now also store 'sex_per_class_acc' and 'diag_per_class_acc' 
    for each classification model.

    Returns a dict with:
      - mae_age, age_true, age_pred
      - acc_sex, sex_true, sex_pred, sex_per_class_acc
      - acc_diag, diag_true, diag_pred, diag_per_class_acc
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, accuracy_score
    
    X_all = np.array(embeddings)
    ages_all = np.array(ages)
    sexes_all = np.array(sexes)
    diags_all = np.array(diags)

    results = {}

    # ~~~~~ DIAGNOSIS (ALL SAMPLES) ~~~~~
    diags_encoded = models["le_diag"].transform(diags_all)
    preds_diag_raw = models["lda_diag"].predict(X_all)
    acc_diag = float(accuracy_score(diags_encoded, preds_diag_raw))
    preds_diag = models["le_diag"].inverse_transform(preds_diag_raw)

    results["acc_diag"] = acc_diag
    results["diag_true"] = diags_all.tolist()
    results["diag_pred"] = preds_diag.tolist()
    # per-class
    diag_per_class_acc = compute_per_class_accuracy(diags_encoded, preds_diag_raw, models["le_diag"])
    results["diag_per_class_acc"] = diag_per_class_acc

    # ~~~~~ AGE & SEX => cognitively_normal only ~~~~~
    normal_mask = (diags_all == "cognitively_normal")
    X_norm = X_all[normal_mask]
    ages_norm = ages_all[normal_mask]
    sexes_norm = sexes_all[normal_mask]

    # ~~~~~ AGE ~~~~~
    if models["reg_age"] is not None and len(X_norm) > 0:
        preds_age = models["reg_age"].predict(X_norm)
        mae_age = float(mean_absolute_error(ages_norm, preds_age))
        results["mae_age"] = mae_age
        results["age_true"] = ages_norm.tolist()
        results["age_pred"] = preds_age.tolist()
    else:
        results["mae_age"] = None
        results["age_true"] = []
        results["age_pred"] = []

    # ~~~~~ SEX ~~~~~
    if models["lda_sex"] is not None and len(X_norm) > 0:
        from sklearn.metrics import accuracy_score
        sexes_norm_encoded = models["le_sex"].transform(sexes_norm)
        preds_sex_raw = models["lda_sex"].predict(X_norm)
        acc_sex = float(accuracy_score(sexes_norm_encoded, preds_sex_raw))
        preds_sex = models["le_sex"].inverse_transform(preds_sex_raw)

        results["acc_sex"] = acc_sex
        results["sex_true"] = sexes_norm.tolist()
        results["sex_pred"] = preds_sex.tolist()

        # per-class for sex
        sex_per_class_acc = compute_per_class_accuracy(sexes_norm_encoded, preds_sex_raw, models["le_sex"])
        results["sex_per_class_acc"] = sex_per_class_acc

    else:
        results["acc_sex"] = None
        results["sex_true"] = []
        results["sex_pred"] = []
        results["sex_per_class_acc"] = {}
    
    return results


def save_per_image_data_split(
    split_name: str,
    split_res: dict,
    base_dir: str,
    save_input_recon: bool = False
):
    """
    Saves per-image data & CSV. 
    If `save_input_recon=False`, or if we don't have input_images_array in split_res,
    we skip writing .npz files for input/recon and leave paths blank.
    """
    import os
    import numpy as np
    import pandas as pd

    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)  # ensure base folder

    # Check if we have input_images_array and recon_images_array
    if "input_images_array" in split_res and "recon_images_array" in split_res:
        input_images_array = split_res["input_images_array"]
        recon_images_array = split_res["recon_images_array"]
        N = input_images_array.shape[0]
    else:
        # We're in a precomputed embeddings scenario => no images
        input_images_array = None
        recon_images_array = None
        # get N from embeddings or any known list
        if "embeddings" in split_res:
            N = split_res["embeddings"].shape[0]
        else:
            # fallback: from one of the metadata lists like ages
            N = len(split_res["ages"])

    # If we do have arrays and user wants to save them
    input_dir = os.path.join(split_dir, "input_images")
    recon_dir = os.path.join(split_dir, "recon_images")
    if save_input_recon and input_images_array is not None:
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

    # We'll always save a .npz with embeddings (if present)
    embeddings_file = os.path.join(split_dir, f"{split_name}_embeddings.npz")
    if "embeddings" in split_res:
        np.savez(embeddings_file, embeddings=split_res["embeddings"])

    records = []
    for i in range(N):
        # If we have input/recon arrays, write them; otherwise leave blank
        if save_input_recon and input_images_array is not None:
            input_path = os.path.join(input_dir, f"input_image_{i+1:06d}.npz")
            recon_path = os.path.join(recon_dir, f"recon_image_{i+1:06d}.npz")
            np.savez(input_path, data=input_images_array[i])
            np.savez(recon_path, data=recon_images_array[i])
        else:
            input_path = ""
            recon_path = ""

        row = {
            "idx": i + 1,
            "input_image_path": input_path,
            "recon_image_path": recon_path,
            # These fields come from arrays or fallback placeholders
            "age": split_res.get("ages", [""]*N)[i],
            "sex": split_res.get("sexes", [""]*N)[i],
            "diagnosis": split_res.get("diags", [""]*N)[i],
            "l1_loss": split_res.get("l1_per_image", [""]*N)[i],
            "perceptual_loss": split_res.get("perceptual_per_image", [""]*N)[i],
            "ssim": split_res.get("ssim_per_image", [""]*N)[i],
            "psnr": split_res.get("psnr_per_image", [""]*N)[i],

            "pred_age": split_res.get("pred_age", [""]*N)[i],
            "pred_sex": split_res.get("pred_sex", [""]*N)[i],
            "pred_diagnosis": split_res.get("pred_diagnosis", [""]*N)[i],
            "pred_age_mae": split_res.get("pred_age_mae", [""]*N)[i],
            "pred_sex_correct": split_res.get("pred_sex_correct", [""]*N)[i],
            "pred_diagnosis_correct": split_res.get("pred_diagnosis_correct", [""]*N)[i],

            "study": split_res.get("study", [""]*N)[i],
            "radiata_id": split_res.get("radiata_id", [""]*N)[i],
        }
        records.append(row)

    df = pd.DataFrame(records)
    csv_file = os.path.join(split_dir, f"{split_name}_metadata.csv")
    df.to_csv(csv_file, index=False)
    print(f"[INFO] Saved {split_name} data to {split_dir}")


def evaluate_autoencoder_on_dataset(
    model, 
    dataset, 
    preprocess_fn,
    device: str = "cpu"
):
    """
    Loop over each record in `dataset`, load the image, do a forward pass,
    accumulate average L1/perceptual, SSIM, PSNR, and collect embeddings + metadata.
    """
    import numpy as np
    from generative.losses import PerceptualLoss
    from torch.nn import L1Loss
    from tqdm import tqdm

    model.eval()
    l1_loss_fn = L1Loss().to(device)
    percep_loss_fn = PerceptualLoss(
        spatial_dims=3,
        network_type="squeeze",
        is_fake_3d=True,
        fake_3d_ratio=0.2
    ).to(device)

    l1_total = 0.0
    percep_total = 0.0
    ssim_total = 0.0
    psnr_total = 0.0
    count = 0

    embeddings_list = []
    ages_list = []
    sexes_list = []
    diags_list = []
    study_list = []
    radiata_list = []

    # Per-image data
    input_images_list = []
    recon_images_list = []
    l1_list = []
    percep_list = []
    ssim_list = []
    psnr_list = []

    for record in tqdm(dataset, total=len(dataset), desc="Evaluating AE"):
        path = record["nii_filepath"]
        meta = record["metadata"]
        
        age = meta["age"]
        sex = meta["sex"]
        diag = meta["clinical_diagnosis"]
        study = meta.get("study", "")
        radiata_id = meta.get("radiata_id", "")

        # 1) Preprocess -> shape (1,1,D,H,W)
        img_tensor = preprocess_fn(path)  # shape (1,1,D,H,W)
        img_tensor = img_tensor.to(device)
        input_images_list.append(img_tensor.cpu().numpy())

        # 2) Forward pass => expect (recon, embeddings, *others)
        with torch.no_grad():
            output = model(img_tensor)

        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise ValueError("Expected model forward pass to return at least 2 items: (recon, embeddings).")

        recon = output[0]           # (1,1,D,H,W)
        embeddings = output[1]      # e.g. (1, latent_dim,...)

        # 3) Compute L1 & Perceptual
        l1_val = l1_loss_fn(recon, img_tensor).item()
        percep_val = percep_loss_fn(recon, img_tensor).item()
        l1_list.append(l1_val)
        percep_list.append(percep_val)
        l1_total += l1_val
        percep_total += percep_val

        # 4) Compute SSIM / PSNR in 3D
        # Convert both to CPU & numpy
        gt_np = img_tensor.cpu().numpy().squeeze()   # shape (D,H,W)
        rc_np = recon.cpu().numpy().squeeze()        # shape (D,H,W)
        
        # If your data is in [0,1], set data_range=1.0
        # For multi-channel or 2D slices, see docs. We'll do simple 3D here:
        ssim_val = ssim(gt_np, rc_np, data_range=1.0)
        psnr_val = psnr(gt_np, rc_np, data_range=1.0)

        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        ssim_total += ssim_val
        psnr_total += psnr_val

        count += 1

        # Save recon
        recon_images_list.append(recon.cpu().numpy())

        # Flatten embeddings
        emb_np = embeddings.view(embeddings.size(0), -1).cpu().numpy()[0]
        embeddings_list.append(emb_np)

        # Collect metadata
        ages_list.append(age)
        sexes_list.append(sex)
        diags_list.append(diag)
        study_list.append(study)
        radiata_list.append(radiata_id)

    # Averages
    avg_l1 = (l1_total / count) if count > 0 else 0.0
    avg_perceptual = (percep_total / count) if count > 0 else 0.0
    avg_ssim = (ssim_total / count) if count > 0 else 0.0
    avg_psnr = (psnr_total / count) if count > 0 else 0.0

    return {
        "avg_l1": avg_l1,
        "avg_perceptual": avg_perceptual,
        "avg_ssim": avg_ssim,
        "avg_psnr": avg_psnr,

        "embeddings": np.array(embeddings_list),
        "ages": ages_list,
        "sexes": sexes_list,
        "diags": diags_list,
        "study": study_list,
        "radiata_id": radiata_list,

        "input_images_array": np.concatenate(input_images_list, axis=0),
        "recon_images_array": np.concatenate(recon_images_list, axis=0),
        "l1_per_image": l1_list,
        "perceptual_per_image": percep_list,
        "ssim_per_image": ssim_list,
        "psnr_per_image": psnr_list
    }


def apply_downstream_predictions(split_res, models):
    """
    For each sample's embedding, store predicted age/sex/diagnosis, plus accuracy checks, etc.
    """
    X = split_res["embeddings"]  # shape (N, latent_dim)

    # We'll create lists
    n = X.shape[0]
    pred_age_list = []
    pred_sex_list = []
    pred_diag_list = []
    pred_age_mae_list = []
    pred_sex_correct_list = []
    pred_diag_correct_list = []

    for i in range(n):
        emb_i = X[i:i+1]  # shape (1, latent_dim)
        true_age = split_res["ages"][i]
        true_sex = split_res["sexes"][i]
        true_diag = split_res["diags"][i]

        # Age prediction => only if cognitively_normal + model is not None
        if true_diag == "cognitively_normal" and models["reg_age"] is not None:
            pred_age_val = float(models["reg_age"].predict(emb_i)[0])
            pred_age_list.append(pred_age_val)
            pred_age_mae_list.append(abs(pred_age_val - true_age))
        else:
            pred_age_list.append("")
            pred_age_mae_list.append("")

        # Sex => only if cognitively_normal + lda_sex
        if true_diag == "cognitively_normal" and models["lda_sex"] is not None:
            numeric_sex = models["le_sex"].transform([true_sex])[0]
            pred_sex_numeric = models["lda_sex"].predict(emb_i)[0]
            pred_sex_val = models["le_sex"].inverse_transform([pred_sex_numeric])[0]
            pred_sex_list.append(pred_sex_val)
            pred_sex_correct_list.append(int(pred_sex_val == true_sex))
        else:
            pred_sex_list.append("")
            pred_sex_correct_list.append("")

        # Diagnosis => always
        diag_numeric = models["le_diag"].transform([true_diag])[0]
        pred_diag_numeric = models["lda_diag"].predict(emb_i)[0]
        pred_diag_val = models["le_diag"].inverse_transform([pred_diag_numeric])[0]
        pred_diag_list.append(pred_diag_val)
        pred_diag_correct_list.append(int(pred_diag_val == true_diag))

    # Attach them to split_res
    split_res["pred_age"] = pred_age_list
    split_res["pred_age_mae"] = pred_age_mae_list
    split_res["pred_sex"] = pred_sex_list
    split_res["pred_sex_correct"] = pred_sex_correct_list
    split_res["pred_diagnosis"] = pred_diag_list
    split_res["pred_diagnosis_correct"] = pred_diag_correct_list


def parse_s3_path(s3_path: str):
    """
    Parse an S3 URI (e.g., 's3://my-bucket/some/prefix') into (bucket, key_prefix).

    Args:
        s3_path (str): An S3 path beginning with 's3://'.

    Returns:
        tuple[str, str]: The (bucket_name, key_prefix) parsed from the given S3 path.

    Raises:
        ValueError: If the path does not start with 's3://'.
    """
    ...
    if not s3_path.startswith("s3://"):
        raise ValueError("Expected s3:// path, got: {s3_path}")
    no_scheme = s3_path[len("s3://"):]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    if len(parts) == 1:
        # e.g. "s3://my-bucket" with no trailing slash
        key_prefix = ""
    else:
        key_prefix = parts[1]
    return bucket, key_prefix


def copy_file_to_s3(local_file: str, s3_path: str):
    """
    Upload a single file to an S3 location using boto3.

    Args:
        local_file (str): The local path of the file to upload.
        s3_path (str): The S3 destination, e.g. 's3://bucket-name/some/prefix/file.ext'.
    """
    ...
    s3 = boto3.client("s3")
    bucket, key_prefix = parse_s3_path(s3_path)
    s3.upload_file(local_file, bucket, key_prefix)
    print(f"Uploaded {local_file} => {s3_path}")


def copy_file_from_s3(s3_path: str, local_file: str):
    """
    Download a single file from an S3 location to the local filesystem.

    Args:
        s3_path (str): The S3 source, e.g. 's3://bucket-name/some/prefix/file.ext'.
        local_file (str): The local file path to save the downloaded object.
    """
    ...
    s3 = boto3.client("s3")
    bucket, key_prefix = parse_s3_path(s3_path)
    s3.download_file(bucket, key_prefix, local_file)
    print(f"Downloaded {s3_path} => {local_file}")


def copy_dir_to_s3(local_dir: str, s3_dir: str):
    """
    Recursively upload files from a local directory to an S3 prefix.

    Args:
        local_dir (str): Path to the local directory containing files to upload.
        s3_dir (str): The destination S3 path, e.g. 's3://bucket-name/prefix/'.

    Note:
        This function currently uploads ALL files/folders in local_dir. If you
        only want certain files (like results from this script), ensure local_dir
        is set to the subfolder containing only the evaluation outputs.
    """
    ...
    s3 = boto3.client("s3")
    bucket, base_prefix = parse_s3_path(s3_dir)

    for root, dirs, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            # relative subpath from local_dir
            rel_path = os.path.relpath(local_path, start=local_dir)
            # join that to the base prefix
            s3_key = os.path.join(base_prefix, rel_path).replace("\\", "/")  # ensure fwd slashes
            s3.upload_file(local_path, bucket, s3_key)
            print(f"Uploaded {local_path} => s3://{bucket}/{s3_key}")


def store_hf_compatible_embeddings(dataset, embeddings, file_path):
    """
    Given the hugging face dataset and the corresponding embeddings (N, embed_dim),
    create a JSON array of records, each with:
      - 'nii_filepath'
      - 'metadata' (from the dataset)
      - 'embedding' (list of floats)
    Then write to file_path.

    This format can be reloaded later to skip inference.
    """
    import json
    import numpy as np

    records = []
    n = len(dataset)
    if embeddings.shape[0] != n:
        raise ValueError(f"Embeddings shape {embeddings.shape} does not match dataset length {n}.")

    for i, record in enumerate(dataset):
        row = {}
        row["nii_filepath"] = record["nii_filepath"]
        row["metadata"] = record["metadata"]
        row["embedding"] = embeddings[i].tolist()  # convert to list of floats
        # Add an 'id' if you like:
        row["id"] = i
        records.append(row)

    with open(file_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[INFO] Saved HF-compatible embeddings to {file_path}")


def load_hf_compatible_embeddings(file_path):
    """
    Loads the JSON file from store_hf_compatible_embeddings.
    Returns a list of dicts. Each dict has 'nii_filepath', 'metadata', 'embedding', etc.
    """
    import json
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} embeddings from {file_path}")
    return data


def build_split_results(ds_split, data_all):
    """
    Create a results dict with 'embeddings', 'ages', 'sexes', 'diags', etc.
    from the precomputed data (data_all).
    We'll match each ds_split item by `radiata_id` (found in metadata).
    
    Requirements:
      - Each item in `data_all` must have: item["metadata"]["radiata_id"] (unique).
      - Each HF dataset record in `ds_split` must have record["metadata"]["radiata_id"].
    """
    import numpy as np

    embeddings_list = []
    ages_list = []
    sexes_list = []
    diags_list = []
    study_list = []
    radiata_list = []

    # Convert data_all into a dict keyed by radiata_id
    lookup = {}
    for item in data_all:
        # Must have item["metadata"]["radiata_id"]
        rid = item["metadata"].get("radiata_id")
        if rid is None:
            raise ValueError("No 'radiata_id' found in data_all item['metadata']. Can't join.")
        # store entire item so we can retrieve 'embedding'
        lookup[rid] = item

    # For each record in ds_split, find matching item in data_all
    for record in ds_split:
        meta = record["metadata"]
        rid = meta.get("radiata_id")
        if rid is None:
            raise ValueError("No 'radiata_id' found in ds_split record['metadata']. Can't match to data_all.")

        item = lookup.get(rid)
        if item is None:
            raise ValueError(f"No precomputed embedding found for radiata_id={rid} in data_all.")

        # gather embedding
        embeddings_list.append(item["embedding"])
        
        # gather typical metadata columns
        age = meta["age"]
        sex = meta["sex"]
        diag = meta["clinical_diagnosis"]
        study = meta.get("study", "")
        # store them
        ages_list.append(age)
        sexes_list.append(sex)
        diags_list.append(diag)
        study_list.append(study)
        radiata_list.append(rid)

    res = {}
    # convert embeddings to float32 array
    res["embeddings"] = np.array(embeddings_list, dtype=np.float32)
    res["ages"] = ages_list
    res["sexes"] = sexes_list
    res["diags"] = diags_list
    res["study"] = study_list
    res["radiata_id"] = radiata_list
    # The script sets L1/percept/etc. to zero or skip them if we're in precomputed mode.

    return res


def combine_and_store_embeddings(ds_train, ds_val, ds_test, train_res, val_res, test_res, embeddings_file):
    """
    Combine train/val/test results into a single HF-compatible JSON
    by calling `store_hf_compatible_embeddings(...)`.
    We'll write to the user-specified `embeddings_file`.
    """
    import numpy as np
    import os
    from datasets import concatenate_datasets

    # Merge them into one dataset
    combined_ds = concatenate_datasets([ds_train, ds_val, ds_test])
    combined_embs = np.concatenate([
        train_res["embeddings"],
        val_res["embeddings"],
        test_res["embeddings"]
    ], axis=0)

    # Handle empty directory (case when embeddings_file has no '/' in it)
    embeddings_dir = os.path.dirname(embeddings_file)
    if not embeddings_dir:  # if it's an empty string
        embeddings_dir = "."

    os.makedirs(embeddings_dir, exist_ok=True)
    store_hf_compatible_embeddings(combined_ds, combined_embs, embeddings_file)


def generate_hf_compatible_embeddings_from_features(
    metadata_csv: str,
    output_file: str
):
    """
    Parse `metadata_csv`, which must have at least a "t1_local_path" column,
    for each row's T1 file. Then:
      1) Derive the IDPs JSON path by replacing the T1 .nii.gz suffix with _IDPs.json,
         or constructing it from the folder structure.
      2) Load the numeric keys from that JSON as the "embedding" (335 values).
      3) Store the rest of the CSV columns in "metadata" except for "t1_local_path".
      4) Write an array of dicts to `output_file` in HF-compatible format:
         [
           {
             "id": <row_index>,
             "nii_filepath": <the t1_local_path>,
             "metadata": {...all other columns...},
             "embedding": [...the numeric IDP values...]
           },
           ...
         ]

    This JSON can be used with `--use_precomputed_embeddings` in model_evals.py,
    skipping the forward pass, but still linking each scan's metadata & IDPs-based embedding.
    """
    import os
    import json
    import pandas as pd
    import collections

    # 1) Read metadata CSV
    df = pd.read_csv(metadata_csv)
    if "t1_local_path" not in df.columns:
        raise ValueError("metadata_csv must contain 't1_local_path' column.")

    records = []
    for i, row in df.iterrows():
        t1_path = row["t1_local_path"]  # e.g. DLBS/sub-XXXX/ses-01/anat/msub-XXXX_ses-01_T1w_brain_affine_mni.nii.gz

        # 2) Construct the IDPs.json path
        # Example logic: replace "_T1w_brain_affine_mni.nii.gz" with "_IDPs.json"
        # Adjust if your naming or folder structure is different.
        if not t1_path.endswith(".nii.gz"):
            raise ValueError(f"Row {i}: 't1_local_path' does not end with .nii.gz => {t1_path}")

        idps_path = t1_path.replace("_T1w_brain_affine_mni.nii.gz", "_IDPs.json")

        # Potentially prepend a root folder if needed, e.g. /Users/jbrown2/company_local/brain-structure/
        # If your CSV paths are already absolute, skip this step
        # For example:
        # root_dir = "/Users/jbrown2/company_local/brain-structure/"
        # idps_path = os.path.join(root_dir, idps_path)

        if not os.path.exists(idps_path):
            raise FileNotFoundError(f"IDPs file not found at {idps_path}")

        # 3) Load the IDPs => 335 numeric values, sorted by key
        with open(idps_path, "r") as f:
            idps_data = json.load(f, object_pairs_hook=collections.OrderedDict)

        embedding_vals = [float(v) for v in idps_data.values()]

        # Convert them to a stable order (alphabetical by key, etc.):
        # idps_data = json.load(f, object_pairs_hook=collections.OrderedDict)
        # embedding_vals = [float(v) for v in idps_data.values()]

        # 4) Build the "metadata" dict from all other CSV columns
        meta = {}
        for c in df.columns:
            if c != "t1_local_path":
                meta[c] = row[c]

        # 5) Construct the final record
        record = {
            "id": i,
            "nii_filepath": t1_path,
            "metadata": meta,
            "embedding": embedding_vals
        }
        records.append(record)

    # 6) Write the resulting array to JSON
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[INFO] Wrote {len(records)} items to {output_file}")


def main():
    """
    Main entry point for model evaluation workflow. Pulls existing results from S3 (optional),
    loads a local HF-style model repo & checkpoint, evaluates on train/val/test splits,
    fits downstream tasks, and optionally pushes results back to S3.
    """
    import argparse
    import json
    import time
    import os
    from datasets import load_dataset
    import joblib

    parser = argparse.ArgumentParser(
        description="Evaluate Brain2Vec model (PCA or VAE) on local or remote data splits."
    )
    parser.add_argument(
        "--model_repo_path", type=str, required=True,
        help="Path to the local cloned model repo directory (e.g., /home/ubuntu/radiata-ai/brain2vec)."
    )
    parser.add_argument(
        "--checkpoint_file", type=str, required=True,
        help="Name of the checkpoint file inside the repo (e.g., autoencoder-ep-4.pth or pca_model.joblib)."
    )
    parser.add_argument(
        "--inference_script", type=str, required=True,
        help="Python script name in model repo for doing inference, eg 'inference_brain2vec.py'."
    )
    parser.add_argument(
        "--pull_model_evals", action="store_true",
        help="If set, pull existing model_evals.json from S3 before appending new results."
    )
    parser.add_argument(
        "--push_files_to_s3", action="store_true",
        help="If set, push the updated model_evals.json and results to S3 after evaluation."
    )
    parser.add_argument(
        "--s3_output_path", type=str, default="s3://radiata-data/leaderboards/brain2vec",
        help="S3 path prefix for storing/fetching results (default: s3://radiata-data/leaderboards/brain2vec)."
    )
    parser.add_argument(
        "--eval_results_dir", type=str, default="eval_results",
        help="Folder name where all evaluation results are stored (NPZ, CSV, etc.). Defaults to 'eval_results'."
    )
    parser.add_argument(
        "--use_precomputed_embeddings",
        action="store_true",
        help="If set, skip model inference. Load embeddings/features from --embeddings_file instead."
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default=None,
        help="Path to load/store embeddings. If None and doing inference, defaults to {eval_results_dir}/embeddings.json."
    )

    args = parser.parse_args()

    # Convert arguments to local variables
    repo_id = args.model_repo_path
    checkpoint_filename = os.path.join(repo_id, args.checkpoint_file)
    inference_module_name = args.inference_script
    pull_model_evals = args.pull_model_evals
    push_files_to_s3 = args.push_files_to_s3
    s3_root = args.s3_output_path

    use_precomputed = args.use_precomputed_embeddings
    eval_results_dir = args.eval_results_dir
    os.makedirs(eval_results_dir, exist_ok=True)

    # We'll place model_evals.json inside eval_results_dir
    results_file = os.path.join(eval_results_dir, "model_evals.json")

    # Decide on embeddings_file if the user didn't provide one
    if not use_precomputed:
        if args.embeddings_file is None:
            # store it in eval_results_dir/embeddings.json
            embeddings_file = os.path.join(eval_results_dir, "embeddings.json")
        else:
            embeddings_file = args.embeddings_file
    else:
        # If user wants precomputed, they'll supply an embeddings_file to load from.
        # If they forgot, we can raise an error or ask them to specify. 
        # But let's just read from args.embeddings_file:
        if not args.embeddings_file:
            raise ValueError("Must provide --embeddings_file if --use_precomputed_embeddings is set.")
        embeddings_file = args.embeddings_file

    print(f"[INFO] Using embeddings_file: {embeddings_file}")

    # Optionally pull existing model_evals.json from S3
    if pull_model_evals:
        s3_model_evals_path = f"{s3_root}/model_evals.json"
        print(f"[INFO] Attempting to pull existing model_evals.json from {s3_model_evals_path} ...")
        try:
            # Temporarily download to a temp name, then place inside eval_results_dir
            tmp_file = "model_evals_temp.json"
            copy_file_from_s3(s3_model_evals_path, tmp_file)
            # Move or rename
            os.replace(tmp_file, results_file)
        except Exception as e:
            print("No existing model_evals.json on S3 or error pulling it:", e)

    # 1) Load model unless we truly skip the entire model usage
    #    Actually, you might still want to load it if 'use_precomputed_embeddings' is True
    #    in case we want to do something else. But if it's not needed, you can skip it.
    if not use_precomputed:
        model, preprocess_fn = load_brain2vec_model(
            repo_id, 
            checkpoint_filename, 
            inference_module_name,
            device=device
        )

    # 2) Load dataset splits
    ds_train = load_dataset("radiata-ai/brain-structure", split="train", trust_remote_code=True)
    ds_val   = load_dataset("radiata-ai/brain-structure", split="validation", trust_remote_code=True)
    ds_test  = load_dataset("radiata-ai/brain-structure", split="test", trust_remote_code=True)

    if use_precomputed:
        # ~~~~~~ Skip inference. Load from embeddings_file ~~~~~~
        print(f"[INFO] Using precomputed embeddings from {embeddings_file}")
        data_all = load_hf_compatible_embeddings(embeddings_file)
        # Now we must separate them into train/val/test by matching them to ds_train, ds_val, ds_test
        # We'll create something like "train_res" that has "embeddings", "ages", "sexes", "diags", etc.

        train_res = build_split_results(ds_train, data_all)
        val_res   = build_split_results(ds_val,   data_all)
        test_res  = build_split_results(ds_test,  data_all)

        # No L1, Perceptual, SSIM, PSNR => set them to 0 or None
        train_res["avg_l1"] = None
        train_res["avg_perceptual"] = None
        train_res["avg_ssim"] = None
        train_res["avg_psnr"] = None

        val_res["avg_l1"] = None
        val_res["avg_perceptual"] = None
        val_res["avg_ssim"] = None
        val_res["avg_psnr"] = None

        test_res["avg_l1"] = None
        test_res["avg_perceptual"] = None
        test_res["avg_ssim"] = None
        test_res["avg_psnr"] = None

    else:
        # ~~~~~~ Normal path: run inference ~~~~~~
        train_res = evaluate_autoencoder_on_dataset(model, ds_train, preprocess_fn, device=device)
        val_res   = evaluate_autoencoder_on_dataset(model, ds_val,   preprocess_fn, device=device)
        test_res  = evaluate_autoencoder_on_dataset(model, ds_test,  preprocess_fn, device=device)

        # Save HF-compatible embeddings if user wants to reuse them later
        # For instance, combine train/val/test into one file or separate them. We'll do one file:
        combine_and_store_embeddings(
            ds_train, ds_val, ds_test,
            train_res, val_res, test_res,
            embeddings_file
        )

    # 4) Fit downstream tasks on train embeddings
    downstream_models = fit_downstream_tasks(
        embeddings_train=train_res["embeddings"],
        ages_train=train_res["ages"],
        sexes_train=train_res["sexes"],
        diags_train=train_res["diags"]
    )

    # Save these downstream models in a subdirectory inside the repo
    model_dir = os.path.join(eval_results_dir, "downstream_models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(downstream_models["reg_age"],  os.path.join(model_dir, "reg_age.pkl"))
    joblib.dump(downstream_models["lda_sex"],  os.path.join(model_dir, "lda_sex.pkl"))
    joblib.dump(downstream_models["le_sex"],   os.path.join(model_dir, "le_sex.pkl"))
    joblib.dump(downstream_models["lda_diag"], os.path.join(model_dir, "lda_diag.pkl"))
    joblib.dump(downstream_models["le_diag"],  os.path.join(model_dir, "le_diag.pkl"))

    print(f"[INFO] Saved downstream models to {model_dir}")

    # Evaluate downstream tasks on all splits
    downstream_train = evaluate_downstream_tasks(downstream_models, train_res["embeddings"], train_res["ages"], train_res["sexes"], train_res["diags"])
    downstream_val   = evaluate_downstream_tasks(downstream_models, val_res["embeddings"],   val_res["ages"],   val_res["sexes"],   val_res["diags"])
    downstream_test  = evaluate_downstream_tasks(downstream_models, test_res["embeddings"],  test_res["ages"],  test_res["sexes"],  test_res["diags"])

    # Attach predictions to the per-image results
    apply_downstream_predictions(train_res, downstream_models)
    apply_downstream_predictions(val_res,   downstream_models)
    apply_downstream_predictions(test_res,  downstream_models)

    # 5) Save per split
    base_dir = eval_results_dir
    os.makedirs(base_dir, exist_ok=True)

    save_per_image_data_split("train", train_res, base_dir, save_input_recon=False)
    save_per_image_data_split("validation", val_res, base_dir, save_input_recon=False)
    save_per_image_data_split("test", test_res, base_dir, save_input_recon=False)

    def summarize_downstream(ds_result):
        return {
            "mae_age": ds_result["mae_age"],
            "acc_sex": ds_result["acc_sex"],
            "sex_per_class_acc": ds_result.get("sex_per_class_acc", {}),
            "acc_diag": ds_result["acc_diag"],
            "diag_per_class_acc": ds_result.get("diag_per_class_acc", {})
        }

    # 6) Summarize all results in model_evals.json
    model_timestamp = int(time.time())
    row_data = {
        "model_name": repo_id,
        "model_hf_path": None,
        "model_nickname": None,
        "checkpoint": checkpoint_filename,
        "checkpoint_hf_path": None,
        "public": False,
        "timestamp": model_timestamp,
        "train_l1": train_res["avg_l1"],
        "train_perceptual": train_res["avg_perceptual"],
        "train_ssim": train_res["avg_ssim"],
        "train_psnr": train_res["avg_psnr"],
        "val_l1": val_res["avg_l1"],
        "val_perceptual": val_res["avg_perceptual"],
        "val_ssim": val_res["avg_ssim"],
        "val_psnr": val_res["avg_psnr"],
        "test_l1": test_res["avg_l1"],
        "test_perceptual": test_res["avg_perceptual"],
        "test_ssim": test_res["avg_ssim"],
        "test_psnr": test_res["avg_psnr"],
        "downstream_train": summarize_downstream(downstream_train),
        "downstream_val": summarize_downstream(downstream_val),
        "downstream_test": summarize_downstream(downstream_test)
    }

    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            json.dump([], f)

    with open(results_file, "r") as f:
        existing = json.load(f)

    existing.append(row_data)

    with open(results_file, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"[INFO] Done! Appended results for model '{repo_id}' to {results_file}")

    # 7) Optionally push updated data to S3
    if push_files_to_s3:
        s3_model_evals_path = f"{s3_root}/model_evals.json"
        print(f"[INFO] Uploading {results_file} to {s3_model_evals_path} ...")
        copy_file_to_s3(results_file, s3_model_evals_path)

        # push the eval outputs
        model_name = repo_id.split('/')[-1]
        s3_results_dir = f"{s3_root}/{model_name}/{eval_results_dir}"
        copy_dir_to_s3(eval_results_dir, s3_results_dir)

    print("[INFO] All done! If pushing to S3, be sure you have AWS credentials configured.")


if __name__ == "__main__":
    main()
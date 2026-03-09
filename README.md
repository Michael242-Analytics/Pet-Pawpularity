# Pawpularity Score Prediction

Predicts a pet photo's popularity score (0–100) using image quality metadata and YOLOv5-extracted features. Built on the [PetFinder.my Pawpularity dataset](https://www.kaggle.com/competitions/petfinder-pawpularity-score) — the motivation being that better photo engagement leads to more adoptions and fewer shelter euthanizations.

---

## What it does

- Loads 9,923 labeled pet photos with manually annotated image quality metadata
- Uses **YOLOv5x6** to detect animals in each photo and compute the bounding-box-to-image area ratio as an engineered feature
- Explores relationships between photo characteristics and pawpularity (blur, proximity, group, species)
- Trains and compares five regression models using 5-fold cross-validation with RMSE
- Evaluates the best model and a top-2 ensemble on a held-out validation set

---

## Dataset

**Source:** PetFinder.my Pawpularity Score (Kaggle)

9,923 pet photos, each labeled with:

| Feature | Description |
|---|---|
| `Focus` | Pet stands out against uncluttered background |
| `Eyes` | Both eyes visible and clear |
| `Face` | Decently clear face, front-facing |
| `Near` | Pet takes up >50% of photo width or height |
| `Action` | Pet is mid-action (jumping, running, etc.) |
| `Accessory` | Physical or digital prop present |
| `Group` | Multiple pets in the photo |
| `Collage` | Photo is a collage |
| `Human` | Human present in photo |
| `Occlusion` | Part of the pet is blocked |
| `Info` | Text or labels overlaid on image |
| `Blur` | Photo is noticeably blurry |
| `Pawpularity` | Target — user engagement score (0–100) |

YOLOv5 adds: `n_pets`, `label` (cat/dog/unknown), `pet_ratio` (bounding box area ÷ image area)

---

## Notebook walkthrough

| Section | Description |
|---|---|
| Data Description | Metadata feature definitions |
| Load & Preview | Read CSV, display top/bottom scoring photos |
| Load YOLOv5 | Load `yolov5x6` model, run inference, extract pet ratios |
| Data Analysis | Correlation heatmap, pawpularity distribution, breakdowns by blur/near/group/species |
| Train/Val Split | 80/20 split, RMSE scoring function |
| CV + GridSearch | 5-fold CV with hyperparameter search across 5 models |
| Decision Tree Viz | Visual printout of top-3 split rules |
| Best Model | Select by val RMSE, plot actual vs predicted |
| Ensemble | Average predictions of top-2 models |
| Takeaway | Findings and limitations |

---

## Models compared

- Ridge Regression (scaled)
- SVR with RBF kernel (scaled)
- Decision Tree
- Random Forest (500 estimators)
- Gradient Boosting

All evaluated via 5-fold cross-validation RMSE. Best single model and a top-2 ensemble are reported.

---

## Key findings

Most metadata features have near-zero linear correlation with pawpularity. `label` (species) shows the strongest correlation (~-0.18), still weak. Pawpularity scores are heavily concentrated in the 20–40 range, which makes prediction outside that range harder.

The takeaway from the project is that image metadata alone has limited predictive power — external signals like posting time or platform context likely account for a significant portion of engagement variance.
****

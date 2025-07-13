AGB and Uncertainty Mapping using XGBoost and Random Forest
This repository contains R scripts and example workflows for modeling Above-Ground Biomass (AGB) and its uncertainty using XGBoost and Random Forest, as applied to high-resolution remote sensing data in three study areas: Beijing, Luochuan, and Zixing.

The code was developed as part of an academic study on spatial patterns of AGB loss and prediction uncertainty under extreme rainfall conditions.

📂 Contents
agb_model.R — Main R script for:

Data preprocessing

Cleaning

AGB prediction with XGBoost

Error modeling (uncertainty) with Random Forest

Raster predictions and output

train_predictions.csv, test_predictions.csv — Sample prediction results

agb_model_metrics.csv, error_model_metrics.csv — Model evaluation metrics

importance_agb.csv, factor_importance_rf.csv — Feature importance from both models

uncertainty_error_rf_model.csv — Uncertainty prediction results

🚀 Workflow

1️⃣ Data Preparation
Input: Point data with AGB and environmental factors from three regions (.csv) and raster predictor layers (.tif).

Remove invalid values (-9999), merge datasets.

2️⃣ AGB Modeling
Train an XGBoost regression model on cleaned point data.

Tune hyperparameters via grid search + cross-validation.

Evaluate performance (R², RMSE, MAE, NSE, PBIAS).

3️⃣ Uncertainty Modeling
Compute residuals of the AGB model.

Train a Random Forest regression model to predict the residuals.

Evaluate performance of the error model.

4️⃣ Raster Prediction
Predict AGB and its uncertainty across the study areas using the trained models and raster predictors.

Save results as GeoTIFF files for further analysis.

🛠️ Requirements
R ≥ 4.0

R packages:

xgboost

randomForest

caret

hydroGOF

terra

dplyr

parallel

You can install the required packages using:
install.packages(c("xgboost", "randomForest", "caret", "hydroGOF", "terra", "dplyr", "parallel"))
📈 Outputs
Best trained models (agb_model.xgb, final_rf_model.rds)

Predicted AGB and uncertainty rasters (predicted_AGB_*.tif, predicted_error_*.tif)

Feature importance tables

Evaluation metrics

📄 Notes
Make sure your file paths in the script correspond to your local data directories.

The script is written to handle large rasters by processing them in chunks.

🔗 Citation
If you use this code in your research, please cite the associated paper:

A framework for assessing aboveground biomass loss caused by rainfall extremes from multisource remote sensing data
Jiaxi Wang, Xiaoqing Luo, Yayi Li, Xinhao Li, Na Deng, Shouzhang Peng, Feng Yang, Jinshi Jiana, Juying Jiao, 2025, 

📬 Contact
For questions, please contact:

Jiaxi Wang

Email: [wangjiaxi@nwafu.edu.cn]

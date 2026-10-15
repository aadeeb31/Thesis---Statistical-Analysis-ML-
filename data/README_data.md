# Data files

`specimens.csv` is the machine-readable master table (176 rows: 160 training specimens, 16 validation specimens; 32 columns). The Excel workbook `SLJ_Auxetic_Repository.xlsx` holds the same specimen table plus configuration summaries, a data dictionary and all derived tables (CV metrics, hyper-parameters, SHAP, validation predictions, failure-mode and configuration-level statistics). `Database_Repository.docx` is a printable description of the same content.

Column definitions are on the `Dictionary` sheet of the workbook. Key points:

* `AI` and `GR` are stored at full precision (6 decimals); the workbook displays 3 decimals.
* `Fmax_N` is the peak lap-shear force; `E_tri_Nmm` = 0.5 · Fmax · δpeak.
* `epoxy_thickness_mm`, `cv_fold`, the out-of-fold predictions and the Shapley columns are blank for validation rows, which were not part of model training.
* Failure-mode labels follow ASTM D5573 (AF, AF-M, CF, CF-IL, CF-M, M, M-B).

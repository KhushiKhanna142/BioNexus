import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

oof = pd.read_parquet('data/serving/oof_predictions.parquet')
pts = pd.read_csv('data/serving/patients.csv')
merged = pd.merge(oof, pts[['patient_id', 'true_pam50']], on='patient_id')

acc = accuracy_score(merged['true_pam50'], merged['pred_label'])
macro_f1 = f1_score(merged['true_pam50'], merged['pred_label'], average='macro')
print('BioNexus OOF recomputed Accuracy:', acc)
print('BioNexus OOF recomputed Macro-F1:', macro_f1)

# Check per fold:
for fold in sorted(merged['fold'].unique()):
    sub = merged[merged['fold'] == fold]
    fold_acc = accuracy_score(sub['true_pam50'], sub['pred_label'])
    print(f"Fold {fold} accuracy: {fold_acc:.4f} (n={len(sub)})")

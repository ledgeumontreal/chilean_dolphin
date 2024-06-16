from sklearn.metrics import confusion_matrix
df = pd.read_csv('/content/Dolphins.csv')
# Example data (replace with your actual data)
y_true = df['Presence']
y_pred_Generalized = df['Generalize']
y_pred_Neuronal = df['Neuronal']
y_pred_RF = df['RF']
# Calculate confusion matrix
tn_Generalized, fp_Generalized, fn_Generalized, tp_Generalized = confusion_matrix(y_true, y_pred_Generalized).ravel()
tn_Neuronal, fp_Neuronal, fn_Neuronal, tp_Neuronal = confusion_matrix(y_true, y_pred_Neuronal).ravel()
tn_RF, fp_RF, fn_RF, tp_RF = confusion_matrix(y_true, y_pred_RF).ravel()
def TSS(tn, fp, fn, tp):
  tss = (tp / (tp + fn)) - (fp / (fp + tn))
  return tss
tss_Generalized = TSS(tn= tn_Generalized, fp = fp_Generalized, fn = fn_Generalized, tp = tp_Generalized)
tss_Neuronal = TSS(tn= tn_Neuronal, fp = fp_Neuronal, fn = fn_Neuronal, tp = tp_Neuronal)
tss_RF = TSS(tn= tn_RF, fp = fp_RF, fn = fn_RF, tp = tp_RF)
print(f"True Skill Statistic (TSS) for Generalized model: {tss_Generalized:.4f}")
print(f"True Skill Statistic (TSS) for Neuronal model: {tss_Neuronal:.4f}")
print(f"True Skill Statistic (TSS) for Random Forest model: {tss_RF:.4f}")
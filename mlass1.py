
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("TOXIC CHEMICAL CLASSIFICATION - MACHINE LEARNING PROJECT")
print("=" * 60)


print("\n" + "=" * 60)
print("STEP 1: LOADING AND EXPLORING DATA")
print("=" * 60)


df = pd.read_csv('data.csv')
print(f"\n✅ Dataset loaded successfully!")
print(f"   - Shape: {df.shape}")
print(f"   - Rows (chemical compounds): {df.shape[0]}")
print(f"   - Columns (features): {df.shape[1]}")


print("\n📊 TARGET VARIABLE DISTRIBUTION:")
target_counts = df['Class'].value_counts()

plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
target_counts.plot(kind='bar', color=colors)
plt.title('Distribution of Toxic vs Non-Toxic Compounds', fontsize=14, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)


for i, v in enumerate(target_counts):
    plt.text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=100)
plt.show()

print("\n" + "=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)


X = df.drop('Class', axis=1)
y = df['Class']

print(f"\n📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")


missing_values = X.isnull().sum().sum()
print(f"\n🔍 Missing values in dataset: {missing_values}")

if missing_values > 0:
    X = X.fillna(X.mean())
    print("✅ Filled missing values with column means")


inf_values = np.isinf(X).sum().sum()
print(f"🔍 Infinite values: {inf_values}")

if inf_values > 0:
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    print("✅ Replaced infinite values")

print(f"\n📊 Data types:")
print(X.dtypes.value_counts())


constant_cols = [col for col in X.columns if X[col].nunique() == 1]
print(f"\n🔍 Constant columns (zero variance): {len(constant_cols)}")

if len(constant_cols) > 0:
    X = X.drop(columns=constant_cols)
    print(f"✅ Dropped {len(constant_cols)} constant columns")
    print(f"   New features shape: {X.shape}")



print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)


print("\n📊 BASIC STATISTICS (first 5 features):")
print(X.iloc[:, :5].describe())


print("\n🔍 Calculating feature correlations with target...")

y_numeric = (y == 'Toxic').astype(int)


correlations = pd.DataFrame({
    'feature': X.columns,
    'correlation': [abs(X[col].corr(y_numeric)) for col in X.columns]
})
correlations = correlations.sort_values('correlation', ascending=False)

print("\n📊 TOP 10 FEATURES CORRELATED WITH TOXICITY:")
print(correlations.head(10))


plt.figure(figsize=(12, 6))
top_corr = correlations.head(15)
sns.barplot(data=top_corr, x='correlation', y='feature', palette='viridis')
plt.title('Top 15 Features Correlated with Toxicity', fontsize=14, fontweight='bold')
plt.xlabel('Absolute Correlation with Target', fontsize=12)
plt.tight_layout()
plt.savefig('feature_correlations.png', dpi=100)
plt.show()


print("\n📊 Visualizing feature distributions by class...")


top_features = correlations.head(4)['feature'].values

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, feature in enumerate(top_features):
    for class_name, color in zip(['NonToxic', 'Toxic'], ['#2ecc71', '#e74c3c']):
        subset = df[df['Class'] == class_name][feature]
        axes[i].hist(subset, alpha=0.7, bins=30, label=class_name, color=color)
    
    axes[i].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=100)
plt.show()


print("\n" + "=" * 60)
print("STEP 4: FEATURE SELECTION")
print("=" * 60)


print("\n🔍 Method 1: Mutual Information Selection")


mi_scores = mutual_info_classif(X, y_numeric, random_state=42)
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(f"\n📊 TOP 10 FEATURES BY MUTUAL INFORMATION:")
print(mi_df.head(10))


k_best = 100
selector_mi = SelectKBest(mutual_info_classif, k=k_best)
X_mi_selected = selector_mi.fit_transform(X, y_numeric)
selected_features_mi = X.columns[selector_mi.get_support()].tolist()

print(f"\n✅ Selected {len(selected_features_mi)} features using Mutual Information")

print("\n🔍 Method 2: Recursive Feature Elimination with CV (this may take a few minutes)...")


rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rfecv = RFECV(estimator=rf, step=50, cv=3, scoring='accuracy', n_jobs=-1, min_features_to_select=50)

if X.shape[0] > 5000:
    X_sample = X.sample(5000, random_state=42)
    y_sample = y_numeric.loc[X_sample.index]
else:
    X_sample = X
    y_sample = y_numeric

rfecv.fit(X_sample, y_sample)
selected_features_rfecv = X.columns[rfecv.support_].tolist()

print(f"\n✅ RFECV selected {len(selected_features_rfecv)} features")
print(f"   Optimal number of features: {rfecv.n_features_}")


plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
         rfecv.cv_results_['mean_test_score'], marker='o')
plt.axvline(x=rfecv.n_features_, color='red', linestyle='--', 
            label=f'Optimal: {rfecv.n_features_} features')
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('RFECV: Accuracy vs Number of Features', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rfecv_results.png', dpi=100)
plt.show()


final_features = list(set(selected_features_mi) & set(selected_features_rfecv))

if len(final_features) < 30:
    print("\n⚠️ Small intersection, using union of methods instead")
    final_features = list(set(selected_features_mi) | set(selected_features_rfecv))

print(f"\n🎯 FINAL SELECTED FEATURES: {len(final_features)}")
print("\nTop 20 selected features:")
print(final_features[:20])


X_selected = X[final_features]
print(f"\n✅ Final features shape: {X_selected.shape}")



print("\n" + "=" * 60)
print("STEP 5: TRAIN-TEST SPLIT AND FEATURE SCALING")
print("=" * 60)


X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training set: {X_train.shape}")
print(f"📊 Testing set: {X_test.shape}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled using StandardScaler")

print("\n" + "=" * 60)
print("STEP 6: MODEL TRAINING AND EVALUATION")
print("=" * 60)


models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Ensemble (Voting)': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ], voting='soft')
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"📈 Training {name}...")
    print(f"{'='*40}")
    
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
   
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
   
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score((y_test == 'Toxic').astype(int), y_pred_proba)
    else:
        auc_score = 'N/A'
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'auc': auc_score,
        'cv_scores': cv_scores
    }
    

    print(f"\n✅ RESULTS FOR {name}:")
    print(f"   - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"   - Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"   - Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"   - AUC-ROC: {auc_score if isinstance(auc_score, str) else f'{auc_score:.4f}'}")
    
  
    print(f"\n   Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))


print("\n" + "=" * 60)
print("STEP 7: MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'CV Accuracy': [results[m]['cv_mean'] for m in results],
    'CV Std': [results[m]['cv_std'] for m in results],
    'Test Accuracy': [results[m]['test_accuracy'] for m in results],
    'AUC-ROC': [results[m]['auc'] if results[m]['auc'] != 'N/A' else 0 for m in results]
})

print("\n📊 MODEL COMPARISON TABLE:")
print(comparison_df.to_string(index=False))


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(results))
width = 0.35

axes[0].bar(x - width/2, [results[m]['cv_mean'] for m in results], width, 
            label='CV Accuracy', color='#3498db', yerr=[results[m]['cv_std']*2 for m in results], capsize=5)
axes[0].bar(x + width/2, [results[m]['test_accuracy'] for m in results], width, 
            label='Test Accuracy', color='#2ecc71')

axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(list(results.keys()), rotation=45, ha='right')
axes[0].legend()
axes[0].set_ylim([0.5, 1.0])
axes[0].grid(axis='y', alpha=0.3)

auc_values = [results[m]['auc'] if results[m]['auc'] != 'N/A' else 0 for m in results]
bars = axes[1].bar(list(results.keys()), auc_values, color=['#3498db', '#e74c3c', '#9b59b6'])
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('AUC-ROC', fontsize=12)
axes[1].set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylim([0.5, 1.0])
axes[1].grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100)
plt.show()


print("\n" + "=" * 60)
print("STEP 8: BEST MODEL ANALYSIS")
print("=" * 60)


best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
best_model = results[best_model_name]['model']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   - Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
print(f"   - CV Accuracy: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std']*2:.4f})")


y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NonToxic', 'Toxic'],
            yticklabels=['NonToxic', 'Toxic'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()


if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': final_features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
    print(feature_importance_df.head(10))

  
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top 15 Most Important Features - {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100)
    plt.show()


print("\n" + "=" * 60)
print("STEP 9: SAVING RESULTS")
print("=" * 60)


with open('selected_features.txt', 'w') as f:
    f.write('\n'.join(final_features))
print("✅ Saved selected features to 'selected_features.txt'")

comparison_df.to_csv('model_comparison.csv', index=False)
print("✅ Saved model comparison to 'model_comparison.csv'")

if hasattr(best_model, 'feature_importances_'):
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    print("✅ Saved feature importance to 'feature_importance.csv'")


print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"""
📊 DATASET SUMMARY:
   - Total compounds: {df.shape[0]}
   - Original features: {df.shape[1] - 1}
   - Selected features: {len(final_features)}
   - Toxic compounds: {target_counts.get('Toxic', 0)}
   - Non-toxic compounds: {target_counts.get('NonToxic', 0)}

🏆 BEST MODEL: {best_model_name}
   - Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}
   - CV Accuracy: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std']*2:.4f})
   - AUC-ROC: {results[best_model_name]['auc'] if results[best_model_name]['auc'] != 'N/A' else 'N/A'}

📈 MODEL PERFORMANCE COMPARISON:
""")

for name, res in results.items():
    print(f"   - {name}: Test Acc = {res['test_accuracy']:.4f}, CV Acc = {res['cv_mean']:.4f}")

print("\n" + "=" * 60)
print("✅ PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
"""
Problem B: Why Did This User Cancel Their Subscription?
Churn Reason Multiclass Classification
Target: price_sensitive | bad_experience | found_alternative | lost_interest
Metric: Macro F1 Score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(r"C:\Users\Devil\Downloads\dummy_B.csv")
test  = pd.read_csv(r"C:\Users\Devil\Downloads\dummy_B.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"\nClass distribution:\n{train['churn_reason'].value_counts()}")
print(f"\nMissing values in train:\n{train.isnull().sum()[train.isnull().sum()>0]}")

# If train and test are the same file (common mistake), create a holdout test split
if train.equals(test):
    print("Warning: train and test are identical — creating a test split from train (80/20).")
    from sklearn.model_selection import train_test_split
    if 'churn_reason' in train.columns:
        train, test = train_test_split(train, test_size=0.2, stratify=train['churn_reason'], random_state=42)
    else:
        train, test = train_test_split(train, test_size=0.2, random_state=42)
    print(f"After split — Train shape: {train.shape}, Test shape: {test.shape}")

# ─── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    # --- Price sensitivity signals ---
    df['price_pain'] = (
        df['num_payment_failures_90d'].fillna(0) * 2 +
        df['price_increase_experienced'].astype(int) * 3 +
        df['used_discount_code'].astype(int) * 1
    )

    # --- Bad experience signals ---
    df['experience_pain'] = (
        df['app_crash_count_30d'].fillna(0) * 2 +
        df['num_support_tickets_90d'].fillna(0) * 1.5 +
        df['unresolved_tickets'].fillna(0) * 3 +
        (df['avg_ticket_resolution_hrs'].fillna(0) / 24).clip(0, 10)
    )
    # Low rating + crashes = bad experience
    df['low_rating_crashes'] = (df['rating_given'] < 2.5).astype(int) * df['app_crash_count_30d'].fillna(0)

    # --- Found alternative signals ---
    df['alt_signal'] = (
        df['competitor_app_installed'].astype(int) * 3 +
        (df['session_trend_30d'] == 'declining').astype(int) * 2 +
        df['num_referrals_made'].fillna(0)  # engaged users switch intentionally
    )

    # --- Lost interest signals ---
    df['disengagement_score'] = (
        df['days_since_last_login'].fillna(0) / 30 +
        df['days_since_last_feature_use'].fillna(0) / 30 +
        (1 - df['notification_opt_in'].astype(int)) * 2 +
        (df['session_trend_30d'] == 'inactive').astype(int) * 3 +
        (df['sessions_per_week'].fillna(0) < 1).astype(int) * 2
    )

    # --- Engagement quality ---
    df['engagement_score'] = (
        df['sessions_per_week'].fillna(0) * 
        df['avg_session_duration_min'].fillna(0) / 100
    )
    df['feature_engagement'] = df['features_used_pct'].fillna(0) / 100

    # --- Subscription value ---
    df['days_per_inr'] = df['subscription_duration_days'] / (df['monthly_price_inr'] + 1)
    df['is_free'] = (df['monthly_price_inr'] == 0).astype(int)
    df['is_long_subscriber'] = (df['subscription_duration_days'] > 365).astype(int)

    # --- Device & connectivity quality ---
    device_quality = {
        'budget_android': 1, 'mid_android': 2, 'flagship_android': 4,
        'iphone': 5, 'ipad': 4, 'desktop': 3, 'multi_device': 5
    }
    df['device_quality_score'] = df['device_type'].map(device_quality).fillna(2)

    net_quality = {
        'slow_2g': 1, 'moderate_3g': 2, 'fast_4g': 4,
        'ultra_5g': 5, 'broadband': 5
    }
    df['net_quality_score'] = df['internet_speed_category'].map(net_quality).fillna(2)
    df['poor_experience_context'] = (df['device_quality_score'] <= 2) & (df['net_quality_score'] <= 2)
    df['poor_experience_context'] = df['poor_experience_context'].astype(int)

    # --- Interactions ---
    df['crash_on_budget'] = df['app_crash_count_30d'].fillna(0) * (df['device_quality_score'] <= 2).astype(int)
    df['price_increase_failure'] = df['price_increase_experienced'].astype(int) * df['num_payment_failures_90d'].fillna(0)
    df['competitor_declining'] = df['competitor_app_installed'].astype(int) * (df['session_trend_30d'] == 'declining').astype(int)
    df['high_usage_then_gone'] = (df['sessions_per_week'].fillna(0) > 5).astype(int) * (df['days_since_last_login'].fillna(0) > 30).astype(int)

    # --- Onboarding ---
    df['onboarding_completed'] = df['onboarding_completed'].astype(int)

    return df


def encode_features(df_train, df_test):
    """Encode categoricals consistently without test leakage."""
    df_train = df_train.copy()
    df_test = df_test.copy()

    cat_cols = [
        'user_age_group', 'gender', 'subscription_plan', 'payment_method',
        'session_trend_30d', 'device_type', 'internet_speed_category',
        'content_category_preference'
    ]
    bool_cols = [
        'competitor_app_installed', 'price_increase_experienced',
        'used_discount_code', 'notification_opt_in', 'onboarding_completed'
    ]

    # Booleans → int (handle missing columns and NaNs safely)
    for col in bool_cols:
        # train
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna(0).astype(int)
        else:
            df_train[col] = 0
        # test
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna(0).astype(int)
        else:
            df_test[col] = 0

    # Label encode categoricals (fit on train only)
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[col] = df_train[col].fillna('unknown').astype(str)
        df_test[col]  = df_test[col].fillna('unknown').astype(str)

        # fit on train only
        le.fit(df_train[col].values)
        df_train[col] = le.transform(df_train[col].values)

        # ensure an 'unknown' class exists for unseen labels in test
        if 'unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'unknown')

        # map unseen test labels to 'unknown' then transform
        known = set(le.classes_)
        df_test[col] = df_test[col].apply(lambda x: x if x in known else 'unknown')
        df_test[col] = le.transform(df_test[col].values)
        encoders[col] = le

    return df_train, df_test

# ─── 3. PREPROCESSING ────────────────────────────────────────────────────────

TARGET = 'churn_reason'
ID_COL = 'id'

# Encode target
label_map = {
    'price_sensitive': 0,
    'bad_experience': 1,
    'found_alternative': 2,
    'lost_interest': 3
}
label_inv = {v: k for k, v in label_map.items()}

train[TARGET] = train[TARGET].map(label_map)

# Drop rows with missing target (if any)
train = train.dropna(subset=[TARGET])
train[TARGET] = train[TARGET].astype(int)

# Feature engineering
train = engineer_features(train)
test  = engineer_features(test)

# Encode
train, test, encoders = encode_features(train, test)

# Fill remaining nulls
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col]  = test[col].fillna(median_val)

FEATURE_COLS = [c for c in train.columns if c not in [ID_COL, TARGET]]

X = train[FEATURE_COLS].values
y = train[TARGET].values
X_test = test[FEATURE_COLS].values
test_ids = test[ID_COL].values

print(f"\nFeature count: {len(FEATURE_COLS)}")
print(f"Class distribution after encode: {np.bincount(y)}")

# ─── 4. CLASS WEIGHTS ────────────────────────────────────────────────────────

classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
cw_dict = dict(zip(classes, class_weights))
print(f"\nClass weights: {cw_dict}")

# ─── 5. MODELS ───────────────────────────────────────────────────────────────

# LightGBM
lgb_params = dict(
    objective='multiclass',
    num_class=4,
    metric='multi_logloss',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced',
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

# XGBoost
xgb_params = dict(
    objective='multi:softprob',
    num_class=4,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
)

# CatBoost
cat_params = dict(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    auto_class_weights='Balanced',
    loss_function='MultiClass',
    eval_metric='TotalF1',
    random_seed=42,
    verbose=0,
)

# ─── 6. OOF + TEST PREDICTIONS (Stacking) ───────────────────────────────────

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb  = np.zeros((len(X), 4))
oof_xgb  = np.zeros((len(X), 4))
oof_cat  = np.zeros((len(X), 4))

test_lgb = np.zeros((len(X_test), 4))
test_xgb = np.zeros((len(X_test), 4))
test_cat = np.zeros((len(X_test), 4))

fold_scores = {'lgb': [], 'xgb': [], 'cat': []}

print("\n" + "="*60)
print("Running 5-Fold CV with LGB + XGB + CatBoost")
print("="*60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
)
    oof_lgb[val_idx]  = lgb_model.predict_proba(X_val)
    test_lgb         += lgb_model.predict_proba(X_test) / N_FOLDS
    score = f1_score(y_val, oof_lgb[val_idx].argmax(1), average='macro')
    fold_scores['lgb'].append(score)
    print(f"  LGB  F1: {score:.4f}")

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=50
    )
    oof_xgb[val_idx]  = xgb_model.predict_proba(X_val)
    test_xgb         += xgb_model.predict_proba(X_test) / N_FOLDS
    score = f1_score(y_val, oof_xgb[val_idx].argmax(1), average='macro')
    fold_scores['xgb'].append(score)
    print(f"  XGB  F1: {score:.4f}")

    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )
    oof_cat[val_idx]  = cat_model.predict_proba(X_val)
    test_cat         += cat_model.predict_proba(X_test) / N_FOLDS
    score = f1_score(y_val, oof_cat[val_idx].argmax(1), average='macro')
    fold_scores['cat'].append(score)
    print(f"  CAT  F1: {score:.4f}")

print("\n" + "="*60)
print("OOF Scores Summary:")
for name, scores in fold_scores.items():
    print(f"  {name.upper():4s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# ─── 7. ENSEMBLE ─────────────────────────────────────────────────────────────

# Simple weighted average (tune weights based on OOF scores)
lgb_mean = np.mean(fold_scores['lgb'])
xgb_mean = np.mean(fold_scores['xgb'])
cat_mean  = np.mean(fold_scores['cat'])
total     = lgb_mean + xgb_mean + cat_mean

w_lgb = lgb_mean / total
w_xgb = xgb_mean / total
w_cat = cat_mean  / total

print(f"\nEnsemble weights — LGB: {w_lgb:.3f}  XGB: {w_xgb:.3f}  CAT: {w_cat:.3f}")

# OOF ensemble
oof_ensemble = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
oof_preds    = oof_ensemble.argmax(1)

oof_f1 = f1_score(y, oof_preds, average='macro')
print(f"\nEnsemble OOF Macro F1: {oof_f1:.4f}")
print("\nClassification Report (OOF):")
print(classification_report(y, oof_preds, target_names=list(label_map.keys())))
print("Confusion Matrix (OOF):")
print(confusion_matrix(y, oof_preds))

# Test ensemble
test_ensemble = w_lgb * test_lgb + w_xgb * test_xgb + w_cat * test_cat
test_preds    = test_ensemble.argmax(1)

# ─── 8. FEATURE IMPORTANCE ───────────────────────────────────────────────────

# Use last fold's LGB model for importance
importance_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Features (LightGBM):")
print(importance_df.head(20).to_string(index=False))

# ─── 9. GENERATE SUBMISSION ──────────────────────────────────────────────────

predictions_str = [label_inv[p] for p in test_preds]

submission = pd.DataFrame({
    'id': test_ids,
    'prediction': predictions_str
})

submission.to_csv("predictions.csv", index=False)
print(f"\nSubmission saved: predictions.csv ({len(submission)} rows)")
print(f"Value counts:\n{submission['prediction'].value_counts()}")

# Validate format
assert set(submission['prediction'].unique()).issubset(set(label_map.keys())), "Invalid label!"
assert submission['id'].nunique() == len(submission), "Duplicate IDs!"
assert submission.isnull().sum().sum() == 0, "Missing values!"
print("\n✓ Submission format validated successfully!")

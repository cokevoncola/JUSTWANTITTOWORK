# tools/normalize_batter_heads.py
import os, joblib

SRC = "models/batter_heads.joblib"
DST = "models/batter_heads.normalized.joblib"

bd = joblib.load(SRC)

# Pick up models from either naming scheme
k_model = bd.get("batter_k_per_pa") or bd.get("k_per_pa")
bip_model = bd.get("contact_shape_head") or bd.get("bip")

k_feats = bd.get("batter_k_features") or bd.get("feature_columns") or []
bip_feats = bd.get("contact_features") or bd.get("feature_columns") or []

# Prefer a single shared feature list if lengths match; otherwise union
if len(k_feats) == len(bip_feats) and k_feats == bip_feats:
    feature_columns = list(k_feats)
else:
    feature_columns = list(dict.fromkeys(list(k_feats) + list(bip_feats)))  # keep order, remove dups

out = {
    "models": {},
    "feature_columns": feature_columns,
    "training_stats": bd.get("meta", {}) or bd.get("training_stats", {}),
    "label_classes": {}
}

if k_model is not None:
    out["models"]["k_per_pa"] = k_model
if bip_model is not None:
    out["models"]["bip"] = bip_model

# If your original bundle included contact classes, keep them
if "classes_contact" in bd:
    out["label_classes"]["bip"] = list(bd["classes_contact"])

joblib.dump(out, DST)
print(f"✅ Wrote {DST}")
print("   models:", list(out["models"].keys()))
print("   n_features:", len(out["feature_columns"]))
if out["label_classes"]:
    print("   label_classes:", out["label_classes"])

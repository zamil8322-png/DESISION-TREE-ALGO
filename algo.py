"""
Decision Tree Practical: ID3 vs C4.5 vs CART  (implemented from scratch)
Dataset: Diabetes prediction dataset (CSV placed in the SAME folder as this script)

Run:  python decision_trees_demo.py
Needs: numpy, pandas, scikit-learn, matplotlib
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# STEP 1: LOAD DATA  (change FILE_NAME to match your csv name)
# ----------------------------------------------------------------------
FILE_NAME = "diabetes_dataset.csv"
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), FILE_NAME)
df = pd.read_csv(path).dropna()
print("Shape:", df.shape)
print(df.head(), "\n")

# target column: 'diabetes' (this dataset), 'Outcome' (Pima), else last column
for cand in ("diabetes", "Outcome"):
    if cand in df.columns:
        TARGET = cand
        break
else:
    TARGET = df.columns[-1]
print("Target column:", TARGET)
print(df[TARGET].value_counts(normalize=True).round(3).to_string(), "\n")

X_raw = df.drop(columns=[TARGET])
y = df[TARGET].astype(int).values

Xr_train, Xr_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.25, random_state=42, stratify=y)
Xr_train, Xr_test = Xr_train.reset_index(drop=True), Xr_test.reset_index(drop=True)

# Numeric version (text columns one-hot encoded) for C4.5 / CART / sklearn
X_train = pd.get_dummies(Xr_train, dtype=float)
X_test = pd.get_dummies(Xr_test, dtype=float).reindex(columns=X_train.columns, fill_value=0.0)
feature_names = list(X_train.columns)

MAX_DEPTH = 4        # keep trees small so they are readable / avoid overfitting
MAX_THRESHOLDS = 32  # candidate split points per feature (keeps big data fast)

# ----------------------------------------------------------------------
# STEP 2: IMPURITY MEASURES
# ----------------------------------------------------------------------
def entropy(y):
    p = np.bincount(y) / len(y)
    p = p[p > 0]
    return -(p * np.log2(p)).sum()

def gini(y):
    p = np.bincount(y) / len(y)
    return 1 - (p ** 2).sum()

# ----------------------------------------------------------------------
# STEP 3: ID3  -> Information Gain, categorical features only, multiway split
#         Numeric columns with many values are first binned into quartiles.
# ----------------------------------------------------------------------
class ID3:
    def __init__(self, max_depth=4, min_samples=5):
        self.max_depth, self.min_samples = max_depth, min_samples

    def fit(self, X, y):
        self.tree = self._build(X, y, list(X.columns), 0)
        return self

    def _build(self, X, y, feats, depth):
        majority = int(np.bincount(y).argmax())
        if len(set(y)) == 1 or not feats or depth >= self.max_depth or len(y) < self.min_samples:
            return {"leaf": majority}
        base = entropy(y)
        best, best_gain = None, 0.0
        for f in feats:
            remainder = 0.0
            for v in X[f].unique():
                m = (X[f] == v).values
                remainder += m.sum() / len(y) * entropy(y[m])
            gain = base - remainder                       # Information Gain
            if gain > best_gain:
                best, best_gain = f, gain
        if best is None or best_gain <= 1e-9:
            return {"leaf": majority}
        node = {"feature": best, "majority": majority, "children": {}}
        rest = [f for f in feats if f != best]
        for v in X[best].unique():
            m = (X[best] == v).values
            node["children"][v] = self._build(X[m], y[m], rest, depth + 1)
        return node

    def _predict_one(self, row):
        node = self.tree
        while "leaf" not in node:
            v = row[node["feature"]]
            if v not in node["children"]:
                return node["majority"]
            node = node["children"][v]
        return node["leaf"]

    def predict(self, X):
        return np.array([self._predict_one(r) for _, r in X.iterrows()])

    def show(self, node=None, indent=""):
        node = node or self.tree
        if "leaf" in node:
            print(indent + f"-> class {node['leaf']}")
            return
        for v, child in node["children"].items():
            print(indent + f"[{node['feature']} = {v}]")
            self.show(child, indent + "    ")

class Discretizer:
    """Bins numeric columns that have many distinct values (needed for ID3).
       Text columns and 0/1 columns are left as they are."""
    def fit(self, X, bins=4):
        self.edges = {}
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 10:
                e = np.unique(X[c].quantile(np.linspace(0, 1, bins + 1)).values)
                e[0], e[-1] = -np.inf, np.inf
                self.edges[c] = e
        return self
    def transform(self, X):
        out = X.copy()
        for c, e in self.edges.items():
            out[c] = pd.cut(X[c], e, labels=False, duplicates="drop")
        return out

# ----------------------------------------------------------------------
# STEP 4: Base class for BINARY trees on numeric features (C4.5 and CART)
# ----------------------------------------------------------------------
class BinaryTree:
    def __init__(self, max_depth=4, min_samples_leaf=5):
        self.max_depth, self.msl = max_depth, min_samples_leaf

    def fit(self, X, y, names):
        self.names = names
        self.tree = self._build(np.asarray(X, float), y, 0)
        return self

    def _best_split(self, X, y):          # overridden by C45 / CART
        raise NotImplementedError

    def _build(self, X, y, depth):
        majority = int(np.bincount(y).argmax())
        if len(set(y)) == 1 or depth >= self.max_depth or len(y) < 2 * self.msl:
            return {"leaf": majority}
        split = self._best_split(X, y)
        if split is None:
            return {"leaf": majority}
        j, t = split
        m = X[:, j] <= t
        return {"feature": j, "threshold": t,
                "left": self._build(X[m], y[m], depth + 1),
                "right": self._build(X[~m], y[~m], depth + 1)}

    def _predict_one(self, x):
        node = self.tree
        while "leaf" not in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["leaf"]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in np.asarray(X, float)])

    def show(self, node=None, indent=""):
        node = node or self.tree
        if "leaf" in node:
            print(indent + f"-> class {node['leaf']}")
            return
        n, t = self.names[node["feature"]], node["threshold"]
        print(indent + f"if {n} <= {t:.3f}:")
        self.show(node["left"], indent + "    ")
        print(indent + "else:")
        self.show(node["right"], indent + "    ")

    # helper: valid (threshold, left_mask) candidates for feature j
    def _candidates(self, X, y, j):
        vals = np.unique(X[:, j])
        if len(vals) > MAX_THRESHOLDS:      # big data: use quantile thresholds
            q = np.quantile(X[:, j], np.linspace(0, 1, MAX_THRESHOLDS + 2)[1:-1])
            thr = np.unique(q)
        else:
            thr = (vals[:-1] + vals[1:]) / 2
        for t in thr:
            m = X[:, j] <= t
            nl = m.sum()
            if nl >= self.msl and len(y) - nl >= self.msl:
                yield t, m

# ----------------------------------------------------------------------
# STEP 5: C4.5 -> Gain RATIO, continuous thresholds, binary split
# ----------------------------------------------------------------------
class C45(BinaryTree):
    def _best_split(self, X, y):
        n, H = len(y), entropy(y)
        per_feature = []                       # (gain, gain_ratio, j, t)
        for j in range(X.shape[1]):
            best = None
            for t, m in self._candidates(X, y, j):
                p = m.sum() / n
                gain = H - p * entropy(y[m]) - (1 - p) * entropy(y[~m])
                split_info = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
                if best is None or gain > best[0]:
                    best = (gain, gain / split_info, j, t)
            if best and best[0] > 1e-9:
                per_feature.append(best)
        if not per_feature:
            return None
        avg_gain = np.mean([c[0] for c in per_feature])
        # Quinlan's heuristic: only consider attributes with gain >= average,
        # then pick the highest GAIN RATIO among them
        good = [c for c in per_feature if c[0] >= avg_gain - 1e-12]
        _, _, j, t = max(good, key=lambda c: c[1])
        return j, t

# ----------------------------------------------------------------------
# STEP 6: CART -> Gini impurity, binary split
# ----------------------------------------------------------------------
class CART(BinaryTree):
    def _best_split(self, X, y):
        n, parent = len(y), gini(y)
        best, best_gain = None, 1e-9
        for j in range(X.shape[1]):
            for t, m in self._candidates(X, y, j):
                p = m.sum() / n
                child = p * gini(y[m]) + (1 - p) * gini(y[~m])
                if parent - child > best_gain:
                    best, best_gain = (j, t), parent - child
        return best

# ----------------------------------------------------------------------
# STEP 7: TRAIN, EVALUATE, COMPARE
# ----------------------------------------------------------------------
def metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (y_true == y_pred).mean(), prec, rec, f1

def report(name, y_true, y_pred):
    acc, prec, rec, f1 = metrics(y_true, y_pred)
    cm = pd.crosstab(pd.Series(y_true, name="Actual"), pd.Series(y_pred, name="Predicted"))
    print(f"\n===== {name} =====")
    print(f"Accuracy {acc:.4f} | Precision {prec:.4f} | Recall {rec:.4f} | F1 {f1:.4f}")
    print(cm)
    return acc, prec, rec, f1

results = {}

# --- ID3
disc = Discretizer().fit(Xr_train)
id3 = ID3(max_depth=MAX_DEPTH).fit(disc.transform(Xr_train), y_train)
results["ID3 (info gain)"] = report("ID3", y_test, id3.predict(disc.transform(Xr_test)))
print("\nID3 tree:"); id3.show()

# --- C4.5
c45 = C45(max_depth=MAX_DEPTH).fit(X_train, y_train, feature_names)
results["C4.5 (gain ratio)"] = report("C4.5", y_test, c45.predict(X_test))
print("\nC4.5 tree:"); c45.show()

# --- CART
cart = CART(max_depth=MAX_DEPTH).fit(X_train, y_train, feature_names)
results["CART (gini)"] = report("CART", y_test, cart.predict(X_test))
print("\nCART tree:"); cart.show()

# --- scikit-learn reference
for crit, label in [("entropy", "sklearn entropy"), ("gini", "sklearn gini")]:
    clf = DecisionTreeClassifier(criterion=crit, max_depth=MAX_DEPTH,
                                 min_samples_leaf=5, random_state=42).fit(X_train, y_train)
    results[label] = metrics(y_test, clf.predict(X_test))
    if crit == "gini":
        plt.figure(figsize=(18, 8))
        plot_tree(clf, feature_names=feature_names, class_names=["No", "Yes"],
                  filled=True, rounded=True, fontsize=8)
        plt.savefig("cart_tree_sklearn.png", dpi=120, bbox_inches="tight")

print("\n===== SUMMARY =====")
print(f"{'Model':22s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
for k, (a, p, r, f) in results.items():
    print(f"{k:22s} {a:7.4f} {p:7.4f} {r:7.4f} {f:7.4f}")
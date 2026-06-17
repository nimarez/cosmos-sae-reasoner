"""Linear-probe baseline for SAE rigor.

A faithful port of the raw-activation logistic-regression baseline from Kantamneni, Engels,
Rajamanoharan, Tegmark & Nanda, "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing"
(arXiv:2502.16681). Before claiming an SAE feature "represents concept X", we check that a cheap
L2-regularized logistic regression on the *raw* residual-stream activations does not already do
the job at least as well. Reported on held-out AUC.

``fit_probe`` takes an arbitrary feature matrix, so a future SAE-feature probe can reuse it
unchanged — only the matrix builder differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # avoid importing the manifest module's heavy transitive deps at import time
    from .manifest import ManifestRecord

# Paper's grid: regularization strength 1/C with C from 1e-5 to 1e5, log-spaced.
DEFAULT_C_GRID: tuple[float, ...] = (
    1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5,
)


@dataclass(frozen=True)
class LinearProbeConfig:
    aggregation: str = "last"  # "last" (paper) | "mean" | "max" — per-record token reduction
    penalty: str = "l2"  # "l2" (paper baseline) | "l1"
    c_grid: tuple[float, ...] = DEFAULT_C_GRID
    max_train: int = 1024  # paper caps training set at 1024 (or dataset max)
    test_frac: float = 0.2
    cv_folds: int = 6  # paper uses 6-fold for 12 < n <= 128
    max_iter: int = 1000
    standardize: bool = True
    seed: int = 0


@dataclass(frozen=True)
class ProbeResult:
    test_auc: float
    test_accuracy: float
    best_c: float
    n_train: int
    n_test: int
    n_features: int
    n_classes: int
    classes: list[str]
    class_counts: dict[str, int]
    train_class_counts: dict[str, int]
    test_class_counts: dict[str, int]
    penalty: str
    standardize: bool
    cv_scheme: str
    selected_feature_ids: list[int] | None = None  # SAE-probe: top-k latents used (None = raw probe)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "test_auc": self.test_auc,
            "test_accuracy": self.test_accuracy,
            "best_c": self.best_c,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "classes": self.classes,
            "class_counts": self.class_counts,
            "train_class_counts": self.train_class_counts,
            "test_class_counts": self.test_class_counts,
            "penalty": self.penalty,
            "standardize": self.standardize,
            "cv_scheme": self.cv_scheme,
        }
        if self.selected_feature_ids is not None:
            out["selected_feature_ids"] = self.selected_feature_ids
        return out


def build_label_map(
    records: Sequence["ManifestRecord"],
    *,
    label_tag: str | None = None,
    label_key: str | None = None,
    label_field: str | None = None,
) -> dict[str, Any]:
    """Resolve a ``record_id -> label`` map from exactly one label source.

    - ``label_tag``: binary presence — ``1`` if the tag is in ``record.tags`` else ``0`` (every
      record is labeled; the zero-relabel exploration path).
    - ``label_key``: ``record.metadata[label_key]`` — records without the key are dropped.
    - ``label_field``: a built-in record attribute (e.g. ``media_type``) — ``None`` values dropped.
    """
    selectors = [s for s in (label_tag, label_key, label_field) if s is not None]
    if len(selectors) != 1:
        raise ValueError("provide exactly one of label_tag, label_key, label_field")

    labels: dict[str, Any] = {}
    for record in records:
        if label_tag is not None:
            labels[record.id] = 1 if label_tag in record.tags else 0
        elif label_key is not None:
            value = (record.metadata or {}).get(label_key)
            if value is not None:
                labels[record.id] = value
        else:  # label_field
            if not hasattr(record, label_field):
                raise ValueError(f"records have no field {label_field!r}")
            value = getattr(record, label_field)
            if value is not None:
                labels[record.id] = value
    return labels


def _group_rows_by_record(
    dataset: Any, label_map: dict[str, Any]
) -> list[tuple[str, list[tuple[int, int]]]]:
    """Ordered ``[(record_id, [(row_index, token_index), ...]), ...]`` for labeled records.

    Deterministic insertion order, so the raw-activation and SAE-feature matrices built from the
    same dataset + label_map align row-for-row (and therefore share an identical train/test split).
    """
    if dataset.activations is None or not dataset.token_coords:
        raise ValueError("activation dataset is empty")
    rows_by_record: dict[str, list[tuple[int, int]]] = {}
    for row, coord in enumerate(dataset.token_coords):
        record = coord.get("record")
        if record is None or record not in label_map:
            continue
        token_index = coord.get("index")
        token_index = int(token_index) if token_index is not None else row
        rows_by_record.setdefault(record, []).append((row, token_index))
    if not rows_by_record:
        raise ValueError("no activation rows matched a labeled record")
    return list(rows_by_record.items())


def _aggregate_block(block: Any, rows: list[tuple[int, int]], aggregation: str) -> Any:
    """Reduce a record's per-token rows (``block`` aligned to ``rows``) to one vector."""
    if aggregation == "last":
        # index of the highest-token-index row within this block
        pos = max(range(len(rows)), key=lambda i: rows[i][1])
        return block[pos]
    if aggregation == "mean":
        return block.mean(dim=0)
    if aggregation == "max":
        return block.amax(dim=0)
    raise ValueError("aggregation must be one of: last, mean, max")


def assemble_probe_matrix(
    dataset: Any,
    label_map: dict[str, Any],
    config: LinearProbeConfig,
) -> tuple[Any, list[Any], list[str]]:
    """Reduce per-token raw activations to one labeled vector per record.

    Aggregates each record's rows per ``config.aggregation`` and attaches the joined label. Returns
    ``(X, y, record_ids)`` where ``X`` is a numpy ``[n_records, hidden_dim]`` array. Records absent
    from ``label_map`` are skipped. Token-subset selection (text-only, assistant-decode-only, a
    single residual stream, …) is done upstream via ``load_activation_dataset(...)``.
    """
    import numpy as np
    import torch

    if config.aggregation not in {"last", "mean", "max"}:
        raise ValueError("aggregation must be one of: last, mean, max")
    acts = dataset.activations
    groups = _group_rows_by_record(dataset, label_map)

    vectors: list[Any] = []
    y: list[Any] = []
    record_ids: list[str] = []
    for record, rows in groups:
        row_idx = torch.tensor([r for r, _ in rows], dtype=torch.long)
        block = acts.index_select(0, row_idx).to(dtype=torch.float32)
        vector = _aggregate_block(block, rows, config.aggregation)
        vectors.append(vector.detach().cpu().numpy())
        y.append(label_map[record])
        record_ids.append(record)

    X = np.stack(vectors, axis=0).astype(np.float32)
    return X, y, record_ids


def assemble_sae_feature_matrix(
    dataset: Any,
    sae: Any,
    label_map: dict[str, Any],
    config: LinearProbeConfig,
    *,
    batch_size: int = 4096,
) -> tuple[Any, list[Any], list[str]]:
    """Like ``assemble_probe_matrix`` but in SAE-feature space.

    Encodes each token's raw activation with ``sae`` (the SAE is non-linear, so we aggregate *after*
    encoding — mean/max-pooling features, or the last token's feature vector). Returns
    ``(X, y, record_ids)`` with ``X`` shaped ``[n_records, feature_dim]``, aligned row-for-row with
    ``assemble_probe_matrix`` over the same dataset + label_map.
    """
    import numpy as np
    import torch

    if config.aggregation not in {"last", "mean", "max"}:
        raise ValueError("aggregation must be one of: last, mean, max")
    acts = dataset.activations
    groups = _group_rows_by_record(dataset, label_map)
    device = next(sae.parameters()).device

    # Each record contributes only the rows its aggregation actually reads: "last" needs a single
    # token (the highest token index), so we skip encoding every other token on the default path.
    if config.aggregation == "last":
        used = [(record, [max(rows, key=lambda pair: pair[1])]) for record, rows in groups]
    else:
        used = [(record, rows) for record, rows in groups]

    # Encode only the rows we need, batched, into a row->feature lookup.
    needed = sorted({r for _, rows in used for r, _ in rows})
    feats_by_row: dict[int, Any] = {}
    with torch.no_grad():
        for start in range(0, len(needed), batch_size):
            chunk = needed[start : start + batch_size]
            idx = torch.tensor(chunk, dtype=torch.long)
            batch = acts.index_select(0, idx).to(device=device, dtype=torch.float32)
            encoded = sae.encode(batch).detach().cpu()
            for pos, row in enumerate(chunk):
                feats_by_row[row] = encoded[pos]

    vectors: list[Any] = []
    y: list[Any] = []
    record_ids: list[str] = []
    for record, rows in used:
        block = torch.stack([feats_by_row[r] for r, _ in rows], dim=0)
        vector = _aggregate_block(block, rows, config.aggregation)
        vectors.append(vector.numpy())
        y.append(label_map[record])
        record_ids.append(record)

    X = np.stack(vectors, axis=0).astype(np.float32)
    return X, y, record_ids


def select_top_k_features(X_train: Any, y_train: Any, k: int) -> Any:
    """Per the paper: the ``k`` latents with the highest mean-absolute class difference on train.

    Binary: ``|mean(class1) - mean(class0)|`` per feature. Multiclass: summed one-vs-rest absolute
    mean differences. Returns feature-column indices (descending score), capped at the column count.
    """
    import numpy as np

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    if len(classes) == 2:
        score = np.abs(X_train[y_train == classes[1]].mean(0) - X_train[y_train == classes[0]].mean(0))
    else:
        score = np.zeros(X_train.shape[1], dtype=np.float64)
        for c in classes:
            score += np.abs(X_train[y_train == c].mean(0) - X_train[y_train != c].mean(0))
    k = min(int(k), X_train.shape[1])
    return np.argsort(score)[::-1][:k]


def _adaptive_cv(n_train: int, minority: int, config: LinearProbeConfig):
    """Paper's size-adaptive validation scheme, capped to keep AUC-scored folds valid.

    n <= 12: leave-two-out-style (many folds); 12 < n <= 128: 6-fold; n > 128: single 80/20 split.
    n_splits is capped by the minority-class count so every validation fold contains both classes.
    """
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    if minority < 2:
        return None, "none (minority class < 2)"
    if n_train <= 12:
        target = max(2, n_train // 2)
        n_splits = min(target, minority)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed), f"leave-two-out~{n_splits}-fold"
    # A single 80/20 holdout needs the minority class to survive the 20% validation slice; with
    # fewer than ~1/0.2 minority examples it can round to zero (nan CV scores), so fall back to
    # minority-capped k-fold instead.
    if n_train > 128 and minority >= 5:
        return (
            StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed),
            "80/20 holdout",
        )
    n_splits = min(config.cv_folds, minority)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed), f"{n_splits}-fold"


def fit_probe(
    X: Any, y: Sequence[Any], config: LinearProbeConfig, *, select_top_k: int | None = None
) -> ProbeResult:
    """Train the paper's logistic-regression probe and report held-out AUC.

    Stratified train/test split, optional in-pipeline standardization (no leakage), GridSearchCV
    over ``C`` scored by AUC, refit best ``C`` on train, evaluate AUC + accuracy on the held-out
    test set. Multiclass uses macro one-vs-rest AUC.

    ``select_top_k`` enables the paper's SAE-feature probe: after the split, pick the top-k feature
    columns by mean-absolute class difference *on the training set only* (no test leakage) and probe
    just those. Leave it ``None`` for the raw-activation baseline.
    """
    import numpy as np
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(list(y))
    classes = [str(c) for c in encoder.classes_]
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError(f"need at least 2 classes to probe; got {n_classes} ({classes})")

    class_counts = {classes[i]: int(c) for i, c in zip(*np.unique(y_enc, return_counts=True))}
    min_overall = min(class_counts.values())
    if min_overall < 2:
        raise ValueError(f"smallest class has {min_overall} example(s); need >= 2 for a train/test split")

    X = np.asarray(X, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=config.test_frac, stratify=y_enc, random_state=config.seed
    )

    # AUC is undefined unless every class appears in the held-out test set; without this guard
    # roc_auc_score silently returns nan (UndefinedMetricWarning). Fail loudly instead.
    if len(np.unique(y_test)) < n_classes:
        missing = [classes[i] for i in range(n_classes) if i not in set(y_test.tolist())]
        raise ValueError(
            f"test split has no examples of class(es) {missing}; task too imbalanced for "
            f"test_frac={config.test_frac} (class counts {class_counts}). Add more rare-class "
            f"examples or raise test_frac."
        )

    # Cap training size (paper: <= 1024), stratified. Skip when the discarded complement would be
    # smaller than n_classes (stratified split can't allocate one per class) — the few extra rows
    # are harmless and the alternative is a ValueError.
    if 0 < config.max_train < len(X_train) and len(X_train) - config.max_train >= n_classes:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=config.max_train, stratify=y_train, random_state=config.seed
        )

    train_counts = Counter(y_train.tolist())
    minority_train = min(train_counts.values())

    # SAE-feature probe: select top-k features on the training set only, then subset both splits.
    selected_feature_ids: list[int] | None = None
    if select_top_k is not None:
        sel = select_top_k_features(X_train, y_train, select_top_k)
        X_train = X_train[:, sel]
        X_test = X_test[:, sel]
        selected_feature_ids = [int(i) for i in sel]

    # sklearn >=1.8 selects the penalty via l1_ratio (0 = L2, 1 = L1); the old penalty= arg is
    # deprecated. liblinear handles L1 efficiently for binary/OvR; lbfgs handles L2.
    if config.penalty == "l1":
        l1_ratio, solver = 1.0, "liblinear"
    else:
        l1_ratio, solver = 0.0, "lbfgs"
    estimator = Pipeline(
        [
            ("scale", StandardScaler() if config.standardize else "passthrough"),
            (
                "clf",
                LogisticRegression(l1_ratio=l1_ratio, solver=solver, max_iter=config.max_iter),
            ),
        ]
    )

    cv, cv_scheme = _adaptive_cv(len(X_train), minority_train, config)
    if cv is None:
        # Too few per-class examples to cross-validate; fall back to a single mid-grid C.
        best_c = 1.0
        estimator.set_params(clf__C=best_c)
        model = estimator.fit(X_train, y_train)
    else:
        scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr"
        search = GridSearchCV(
            estimator,
            {"clf__C": list(config.c_grid)},
            scoring=scoring,
            cv=cv,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_c = float(search.best_params_["clf__C"])
        model = search.best_estimator_

    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)
    if n_classes == 2:
        test_auc = float(roc_auc_score(y_test, proba[:, 1]))
    else:
        test_auc = float(roc_auc_score(y_test, proba, multi_class="ovr", labels=np.arange(n_classes)))
    test_accuracy = float(accuracy_score(y_test, preds))

    def _named(counts: dict[int, int] | Counter) -> dict[str, int]:
        return {classes[int(k)]: int(v) for k, v in counts.items()}

    return ProbeResult(
        test_auc=test_auc,
        test_accuracy=test_accuracy,
        best_c=best_c,
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        n_features=int(X_train.shape[1]),  # post-selection feature count actually probed
        n_classes=n_classes,
        classes=classes,
        class_counts=class_counts,
        train_class_counts=_named(train_counts),
        test_class_counts=_named(Counter(y_test.tolist())),
        penalty=config.penalty,
        standardize=config.standardize,
        cv_scheme=cv_scheme,
        selected_feature_ids=selected_feature_ids,
    )

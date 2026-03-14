import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)


def generate_data(n_samples: int=5000, incident_rate: float=0.05, incident_length: int=20):
    time = np.array(n_samples)

    # sin signals with some noise
    cpu = 40 + 5 * np.sin(2*np.pi*time/500) + np.random.normal(0, 2, n_samples)
    memory = 55 + 3 * np.sin(2*np.pi*time/700) + np.random.normal(0, 1.5, n_samples)
    latency = 100 + 10 * np.sin(2*np.pi*time/300) + np.random.normal(0, 5, n_samples)

    # incidents
    incident_flag = np.zeros(n_samples, dtype=int)
    n_incidents = int(n_samples * incident_rate / incident_length)
    starts = sorted(np.random.choice(np.arange(50, n_samples-incident_length-50), size=n_incidents, replace=False))
    # no overlap
    filtered_starts = []
    last_end = -100
    for s in starts:
        if s > last_end + 10:
            filtered_starts.append(s)
            last_end = s + incident_length

    for f in filtered_starts:
        end = f + incident_length
        incident_flag[f:end] = 1

        # memory rises 20 steps before
        pre = f - 10
        memory[pre:f] += np.linspace(0, 15, 10)

        # incident anomaly
        ramp = np.linspace(0, 1, incident_length)
        cpu[f:end] += 35*ramp + np.random.normal(0, 3, incident_length)
        memory[f:end] += 20*ramp + np.random.normal(0, 2, incident_length)
        latency[f:end] += 200*ramp + np.random.normal(0, 15, incident_length)
    
    # realistic ranges
    cpu = np.clip(cpu, 0, 100)
    memory = np.clip(memory, 0, 100)
    latency = np.clip(latency, 0, 600)

    return pd.DataFrame({'cpu': cpu, 'memory': memory, 'latency': latency, 'incident': incident_flag})

def sliding_window(data: pd.DataFrame, W: int, H: int):
    metrics = ['cpu', 'memory', 'latency']
    values = data[metrics].values
    flags = data['incident'].values

    X_list, y_list = [], []

    for i in range(len(data) - W - H + 1):
        window = values[i:i+W]
        future = flags[i+W:i+W+H]
        label = int(future.max() > 0)

        feats = []
        for j in range(3):
            series = window[:,j]
            feats += [
                series.mean(),
                series.std(),
                series.min(),
                series.max(),
                series[-1],
                series[-1] - series[0],
                np.polyfit(np.arange(W), series, 1)[0],
                np.polyfit(np.arange(5), series[-5:], 1)[0],
            ]    
        
        cpu_memory = np.corrcoef(window[:,0], window[:,1])[0,1]
        cpu_latency = np.corrcoef(window[:,0], window[:,2])[0,1]
        feats += [cpu_memory, cpu_latency]

        X_list.append(feats)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    feat_names = []
    for m in metrics:
        feat_names += [f'{m}_mean', f'{m}_std', f'{m}_min', f'{m}_max',
                       f'{m}_last', f'{m}_delta', f'{m}_slope', f'{m}_slope5']
    feat_names += ['corr_cpu_mem', 'corr_cpu_lat']

    return X, y, feat_names


# split train/val/test 
def temporal_split(X, y, train_frac=0.6, val_frac=0.2):
    n = len(y)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return (X[:i1], y[:i1],
            X[i1:i2], y[i1:i2],
            X[i2:], y[i2:])


# training
def train_models(X_tr, y_tr):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=10,
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, 
            max_depth=4,
            learning_rate=0.05, 
            subsample=0.8,
            random_state=42),
    }

    trained = {}
    for name, clf in models.items():
        clf.fit(X_tr_s, y_tr)
        trained[name] = clf
        print(f"[Train] {name}")

    return trained, scaler


def best_f1_threshold(y_true, proba):
    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1)
    return thresholds[best_idx], f1[best_idx]

# evaulation
def evaluate(name, clf, scaler, X_val, y_val, X_te, y_te):
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    prob_val = clf.predict_proba(X_val_s)[:, 1]
    prob_test = clf.predict_proba(X_te_s)[:, 1]

    # threshold from validation set
    thresh, _ = best_f1_threshold(y_val, prob_val)

    pred_te = (prob_test >= thresh).astype(int)

    roc = roc_auc_score(y_te, prob_test)
    pr = average_precision_score(y_te, prob_test)
    f1 = f1_score(y_te, pred_te)
    cm = confusion_matrix(y_te, pred_te)

    print(f"{name}:")
    print(f"Alert threshold (val F1-max) : {thresh:.3f}")
    print(f"ROC-AUC   : {roc:.4f}")
    print(f"PR-AUC    : {pr:.4f}")
    print(f"F1 (test) : {f1:.4f}")
    print(classification_report(y_te, pred_te, target_names=['Normal', 'Incident']))
    return {
        'name': name, 'clf': clf,
        'prob_val': prob_val, 'prob_test': prob_test,
        'thresh': thresh, 'roc': roc, 'pr': pr, 'f1': f1,
        'cm': cm, 'pred_te': pred_te,
    }


# visualisation
def plot_all(df, results, feat_names, y_te):

    colors_metrics = ['#1976D2', '#43A047', '#E53935']
    labels_metrics = ['CPU (%)', 'Memory (%)', 'Latency (ms)']
    col_colors = {'Random Forest': '#2196F3', 'Gradient Boosting': '#FF5722'}
    col_cmaps  = {'Random Forest': 'Blues',   'Gradient Boosting': 'Oranges'}
    fig = plt.figure(figsize=(22, 42))
    gs  = gridspec.GridSpec(6, 2, figure=fig, hspace=0.52, wspace=0.35,
                            height_ratios=[1.1, 1, 1, 1, 1.4, 1])

    ax_ts = fig.add_subplot(gs[0, :])
    for col, lab, c in zip(['cpu', 'memory', 'latency'], labels_metrics, colors_metrics):
        norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        ax_ts.plot(norm, label=lab, color=c, lw=0.7, alpha=0.8)
    inc = df['incident'].values
    in_inc = False
    for i in range(len(inc)):
        if inc[i] == 1 and not in_inc:
            start = i; in_inc = True
        elif inc[i] == 0 and in_inc:
            ax_ts.axvspan(start, i, color='red', alpha=0.12)
            in_inc = False
    ax_ts.legend(handles=[
        *[plt.Line2D([0],[0], color=c, lw=2, label=l)
          for c, l in zip(colors_metrics, labels_metrics)],
        Patch(facecolor='red', alpha=0.25, label='Incident')], loc='upper right')
    ax_ts.set_title('Synthetic Time-Series Data (normalised) with Incident Regions',
                    fontsize=13, fontweight='bold')
    ax_ts.set_xlabel('Time step'); ax_ts.set_ylabel('Normalised value')
    ax_ts.set_xlim(0, len(df))

    for col_idx, r in enumerate(results):
        name  = r['name']
        color = col_colors[name]
        cmap  = col_cmaps[name]

        # PR curve
        ax_pr = fig.add_subplot(gs[1, col_idx])
        p, rec, _ = precision_recall_curve(y_te, r['prob_test'])
        ax_pr.plot(rec, p, color=color, lw=2)
        ax_pr.set_title(f'PR Curve – {name}\n(AP={r["pr"]:.3f})', fontweight='bold')
        ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
        ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1); ax_pr.grid(alpha=0.3)

        # ROC curve
        ax_roc = fig.add_subplot(gs[2, col_idx])
        fpr, tpr, _ = roc_curve(y_te, r['prob_test'])
        ax_roc.plot(fpr, tpr, color=color, lw=2)
        ax_roc.set_title(f'ROC Curve – {name}\n(AUC={r["roc"]:.3f})', fontweight='bold')
        ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
        ax_roc.grid(alpha=0.3)

        # Confusion matrix
        ax_cm = fig.add_subplot(gs[3, col_idx])
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap=cmap,
                    xticklabels=['Normal', 'Incident'],
                    yticklabels=['Normal', 'Incident'],
                    ax=ax_cm, linewidths=0.5, cbar=False, annot_kws={'size': 14})
        tn, fp, fn, tp = r['cm'].ravel()
        ax_cm.set_title(
            f'Confusion Matrix – {name}\n'
            f'TP={tp}  FP={fp}  FN={fn}  TN={tn}  |  F1={r["f1"]:.3f}',
            fontweight='bold')
        ax_cm.set_ylabel('True label'); ax_cm.set_xlabel('Predicted label')

        
        # F1 curve
        ax_f1 = fig.add_subplot(gs[4, col_idx])
        prec_v, rec_v, thresh_v = precision_recall_curve(y_te, r['prob_test'])
        f1_curve = 2 * prec_v[:-1] * rec_v[:-1] / (prec_v[:-1] + rec_v[:-1] + 1e-9)
        ax_f1.plot(thresh_v, f1_curve, color=color, lw=2)
        best_f1_val = f1_curve[np.argmax(f1_curve)]
        ax_f1.set_title(f'F1 Curve – {name}', fontweight='bold')
        ax_f1.set_xlabel('Threshold'); ax_f1.set_ylabel('F1 Score')
        ax_f1.set_xlim(0, 1); ax_f1.set_ylim(0, 1)
        ax_f1.legend(fontsize=9); ax_f1.grid(alpha=0.3)

        # Score distribution
        ax_dist = fig.add_subplot(gs[5, col_idx])
        c_normal   = '#42A5F5' if col_idx == 0 else '#FFB74D'
        c_incident = '#E53935' if col_idx == 0 else '#BF360C'
        for cls, c_hist, lab in [(0, c_normal, 'Normal'), (1, c_incident, 'Incident')]:
            mask = y_te == cls
            ax_dist.hist(r['prob_test'][mask], bins=40, alpha=0.55,
                         color=c_hist, label=lab, density=True)
        ax_dist.axvline(r['thresh'], color='black', lw=1.5,
                        linestyle='--', label=f'Threshold={r["thresh"]:.2f}')
        ax_dist.set_title(f'Score Distribution – {name}', fontweight='bold')
        ax_dist.set_xlabel('Predicted probability of incident')
        ax_dist.set_ylabel('Density')
        ax_dist.legend(fontsize=9); ax_dist.grid(alpha=0.3)


    plt.suptitle('Incident Prediction – Sliding Window Classification',
                 fontsize=16, fontweight='bold', y=1.005)
    out = 'incident_prediction_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out}")
    plt.close()


def main():
    W, H = 30, 10

    # generating data
    data = generate_data()
    
    # building sliding window
    X, y, feat_names = sliding_window(data, W, H)
    
    # split
    X_tr, y_tr, X_val, y_val, X_te, y_te = temporal_split(X, y)
    print(f"        incident rate: train={y_tr.mean():.3f}  "
          f"val={y_val.mean():.3f}  test={y_te.mean():.3f}")

    # train
    trained_models, scaler = train_models(X_tr, y_tr)

    # evaluate
    results = []
    for name, clf in trained_models.items():
        r = evaluate(name, clf, scaler, X_val, y_val, X_te, y_te)
        results.append(r)

    # visualise
    plot_all(data, results, feat_names, y_te)
    
    

if __name__ == '__main__':
    main()
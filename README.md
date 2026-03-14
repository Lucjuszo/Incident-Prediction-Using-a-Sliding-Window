# Incident-Prediction-Using-a-Sliding-Window
Lucjan Butko

Implement a model that predicts whether an incident will occur within the next H time steps based on the previous W steps of one or more time-series metrics. Use a sliding-window formulation and train the model using any standard machine-learning framework.

The applicant may use any suitable public dataset or generate a synthetic time series with labeled incident intervals (e.g. anomalies or threshold breaches). The emphasis is on correct problem formulation, model selection, training, and evaluation rather than dataset complexity or model size.

The solution should include a clear description of the modeling choices, the evaluation setup (including alert thresholds and metrics), and an analysis of the results. During follow-up, the applicant should be able to explain the design decisions, discuss limitations, and outline how the approach could be adapted to a real alerting system.

# Results 
### Random Forest
| Metric | Value |
|:-------|------:|
| Alert Threshold (val F1-max) | 0.184 |
| ROC-AUC | 0.9735 |
| PR-AUC | 0.9334 |
| F1 (test) | 0.9080 |

| Class | Precision | Recall | F1-Score | Support |
|:------|----------:|-------:|---------:|--------:|
| Normal | 0.99 | 0.99 | 0.99 | 906 |
| Incident | 0.91 | 0.91 | 0.91 | 87 |
| accuracy | | | 0.98 | 993 |
| macro avg | 0.95 | 0.95 | 0.95 | 993 |
| weighted avg | 0.98 | 0.98 | 0.98 | 993 |

### Gradient Boosting
| Metric | Value |
|:-------|------:|
| Alert Threshold (val F1-max) | 0.591 |
| ROC-AUC | 0.9832 |
| PR-AUC | 0.9454 |
| F1 (test) | 0.9461 |

| Class | Precision | Recall | F1-Score | Support |
|:------|----------:|-------:|---------:|--------:|
| Normal | 0.99 | 1.00 | 1.00 | 906 |
| Incident | 0.99 | 0.91 | 0.95 | 87 |
| accuracy | | | 0.99 | 993 |
| macro avg | 0.99 | 0.95 | 0.97 | 993 |
| weighted avg | 0.99 | 0.99 | 0.99 | 993 |

# Report 
### Objective
Build a model that, given the last W=30 time steps of three system metrics (CPU, memory, latency), predicts whether an incident will occur within the next H=10 time steps.

### Data
A synthetic time series of 5000 time steps was generated with 12 injected incidents (incident ratio = 4.8%). Each incident lasts 20 steps and follows a realistic pattern:
Gradual memory rise 10 steps before the incident,
Sharp CPU and latency spike during the incident.

Data was split chronologically into training (60%), validation (20%) and test (20%) sets — without random shuffling, which prevents future data from leaking into training.

### Role of Each Metric
Memory — the leading signal. It rises gradually 10 steps before the incident, simulating e.g. a memory leak. This is the most important feature according to the model (memory_slope ranked first) — the model learned that a rising memory trend is an early warning sign.
CPU — the concurrent signal. Spikes sharply once the incident is already underway. High CPU alone is not alarming, but combined with rising memory it becomes a strong indicator.
Latency — the most dramatic signal. Jumps by 200ms during the incident (largest amplitude of all metrics). It is the most noticeable symptom for end users — server response time degrades from 100ms to 300ms.

### Plot Analysis
PR Curve — both curves maintain high precision up to Recall ~0.85, then drop sharply. This means the models are confident in their predictions for most incidents, and only the hardest cause difficulty.
ROC Curve – Both curves rise to values ​​close to 1 (AUC ~0.97–0.98), confirming that both models reliably distinguish normal operation from incidents.
Confusion Matrix — both models made 8 FN errors - missed incidents. Random Forest additionally produced 8 FP errors - false alarms, while Gradient Boosting produced only 1.
F1 Curve — Random Forest shows a wide plateau, meaning it is robust to threshold selection. Such a broad and high peak of the curve indicates that the model is very confident in its predictions.
Score Distribution — both models clearly separate the two classes, indicating high prediction confidence. Gradient Boosting separates the classes almost perfectly.

## Conclusions
Gradient Boosting is the better choice for this task — higher F1, near-zero false alarms (FP=1) and superior PR-AUC.
The main limitation is the synthetic nature of the data — real system metrics are noisier and incidents are more irregular.

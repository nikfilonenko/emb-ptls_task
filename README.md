# About the project ***`emb-ptls_task`***

#### Решение задания: [ТУТ](src/notebooks/tech_task_coles_emb.ipynb)

- PandasDataPreprocessor

```python
[{'customer_id': 14,
  'transaction_date': tensor([ 14,  69,  72,  79, 106, 217, 222, 232, 266, 273]),
  'event_time': tensor([ 14,  69,  72,  79, 106, 217, 222, 232, 266, 273]),
  'product_category': tensor([1, 3, 4, 4, 1, 2, 4, 4, 2, 3]),
  'quantity': tensor([1, 8, 6, 1, 2, 5, 6, 9, 8, 4]),
  'price': tensor([96.4300, 54.1800, 40.5900, 70.2600, 23.7800, 60.6500, 92.3200, 26.9400, 23.8000, 26.5600], dtype=torch.float64),
  'discount_applied': tensor([ 5., 21., 12., 20., 24., 16.,  8., 16.,  5., 27.], dtype=torch.float64),
  'total_amount': tensor([ 91.6100, 342.4200, 214.3200,  56.2100,  36.1500, 254.7300, 509.6100, 203.6700, 180.8800,  77.5600], dtype=torch.float64)
}]
```

- CoLES

INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=15` reached.

valid/recall_top_k: `0.855`

loss: `136.579`

![image](https://github.com/user-attachments/assets/d132c176-c58a-4ace-8b84-ca475178a8d8)

- Random forest training based on received embeddings

**Experiment 1: Stratification cross-validation**

```python
Best parameters found:  {'max_depth': None, 'n_estimators': 200}
Test Accuracy: 0.7735
              precision    recall  f1-score   support

           0       0.74      0.94      0.83     11123
           1       0.87      0.62      0.73      5556
           2       0.84      0.33      0.47      2364

    accuracy                           0.77     19043
   macro avg       0.82      0.63      0.68     19043
weighted avg       0.79      0.77      0.76     19043
```

**Experiment 2: Class weight**

```python
Test Accuracy: 0.7834

              precision    recall  f1-score   support

           0       0.76      0.92      0.83     11123
           1       0.87      0.63      0.73      5556
           2       0.77      0.49      0.60      2364

    accuracy                           0.78     19043
   macro avg       0.80      0.68      0.72     19043
weighted avg       0.79      0.78      0.77     19043
```

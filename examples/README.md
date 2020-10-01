
# Example Jupyter Notebooks

This folder will have examples on how metriks can be used for calculating precision and recall on ranking models.
Please create subfolders for each example depending on your dataset.

Steps included in all examples are,

1. A default dataset has been choosen.
2. Trained a ranking model with the dataset choosen.
3. Finally demonstrated how these metrics can be used on our dataset,

| Python API                                                 | Description                                                                   |
+============================================================+===============================================================================+
| `metriks.recall_at_k(y_true, y_prob, k)`                   | Calculates recall at k for binary classification ranking problems.            |
+------------------------------------------------------------+-------------------------------------------------------------------------------+
| `metriks.precision_at_k(y_true, y_prob, k)`                | Calculates precision at k for binary classification ranking problems.         |
+------------------------------------------------------------+-------------------------------------------------------------------------------+
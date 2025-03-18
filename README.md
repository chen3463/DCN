                            +-------------------------+
                            |     Input Layer         |
                            | (Numerical, Categorical |
                            |     Embeddings,         |
                            |    One-Hot Features)    |
                            +-----------+-------------+
                                        |
                            +-----------v-------------+
                            | Embedding Layer         |
                            | (Categorical Features)  |
                            +-----------+-------------+
                                        |
                        +---------------v----------------+
                        | Concatenate All Features      |
                        | (Numerical + Embeddings + One|
                        | Hot Features)                |
                        +---------------+----------------+
                                        |
                    +-------------------+-------------------+
                    |                                       |
        +-----------v-----------+               +-----------v-----------+
        |  Cross Network         |               |   Deep Network        |
        |  (Feature Interactions)|               |   (High-Level         |
        |                        |               |    Abstractions)      |
        +-----------+------------+               +-----------+-----------+
                    |                                       |
              +-----v-----+                           +-----v-----+
              |   Cross   |                           |   Deep    |
              |   Output  |                           |   Output  |
              +-----+-----+                           +-----+-----+
                    |                                       |
                    +-------------------+-------------------+
                                        |
                           +------------v-------------+
                           |   Concatenate Outputs   |
                           |  (Cross + Deep Network) |
                           +------------+-------------+
                                        |
                           +------------v-------------+
                           |   Output Layer (Sigmoid) |
                           +------------+-------------+
                                        |
                           +------------v-------------+
                           |      Final Output       |
                           |     (Probability)       |
                           +-------------------------+


```
project/
│
├── data_preprocessing.py      # Handles data preprocessing and dataset creation
├── model.py                   # Contains the model architecture (DCNv2)
├── train.py                   # Handles training and evaluation logic
├── hyperparameter_optimization.py  # Contains the Optuna optimization loop
├── feature_importance.py      # Contains feature importance calculation using SHAP
├── utils.py                   # Utility functions (e.g., data preparation, feature names)
└── main.py                    # Main script to run the pipeline
```


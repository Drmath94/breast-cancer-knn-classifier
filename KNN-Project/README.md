
# Predicting Breast Cancer Diagnosis Using K-Nearest Neighbors (KNN)

# Project Description:

This project focused on predicting whether a tumor is benign or malignant using the Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository. The goal was to build a robust classification model that could aid in early breast cancer detection.

Key steps and processes included:

🔍 Data Exploration and Preprocessing:

Performed initial data cleaning and normalization.

Scaled the dataset features to improve the distance-based performance of KNN.

Used np.ascontiguousarray to optimize the scaled data (Xscaled_df) for better computational efficiency.

🧠 Model Training and Hyperparameter Tuning:

Implemented a K-Nearest Neighbors (KNN) classifier.

Evaluated the model using 10-fold cross-validation to ensure generalization and reduce variance in performance.

Conducted hyperparameter tuning by iterating over a range of n_neighbors values (1 to 49) to identify the best-performing configuration.

📊 Performance Evaluation:

Calculated the mean cross-validation score for each value of n_neighbors.

Analyzed the resulting scores (scores_1) to determine the optimal number of neighbors for the KNN model.

The final model achieved an impressive 95.91% test accuracy, indicating strong generalization to unseen data.

🧾 Classification Report Interpretation:

For malignant tumors (M), the model achieved a precision of 0.97, recall of 0.90, and an F1-score of 0.93, meaning it accurately detected most malignant cases but had a few false negatives.

For benign tumors (B), it achieved a precision of 0.95, recall of 0.98, and an F1-score of 0.96, demonstrating excellent performance in correctly identifying non-cancerous cases.

The overall accuracy of the model on the test set was 95%, with a macro-average F1-score of 0.95, showing balanced performance across both classes.

The weighted average scores confirm consistent high-quality predictions, especially important in medical diagnostics.

![Logo](https://th.bing.com/th/id/OIP.0DpWA8ngyJd44YTfbmds7wHaE8?w=272&h=181&c=7&r=0&o=5&dpr=1.3&pid=1.7)
## Authors

- Hello, I'm Emmanuel Oladele


## 🚀 About Me

Highly motivated and detail-driven Data Scientist, ML Engineer, and NLP Specialist
## Installation

Install my-project with

```bash
  import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # to plot
import seaborn as sns # to plot
from pyforest import *


%matplotlib inline

```
    
## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## 🛠 Skills
Programming Language: Python 

Frameworks : Pandas,Numpy,Seaborn,Matplotlib,Sklearn

Platform : Jupyter Notebook a

soft skill: Problem solving skills, Time Management, Communication Skills.


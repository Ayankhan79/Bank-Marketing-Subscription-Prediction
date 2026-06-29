# 🏦 Bank Marketing Subscription Prediction

This project predicts whether a customer will subscribe to a **term deposit** using the Bank Marketing dataset. Multiple gradient boosting algorithms—**XGBoost, LightGBM, and CatBoost**—were trained and compared to identify the best-performing model. Since the dataset is highly imbalanced, the models were optimized using **RandomizedSearchCV** with **5-fold Stratified Cross-Validation**, and evaluated primarily using **F1-Score** and **ROC-AUC**.

## 🚀 Project Highlights

* Preprocessed and cleaned the dataset.
* Handled categorical variables and class imbalance.
* Tuned **XGBoost, LightGBM, and CatBoost** using `RandomizedSearchCV`.
* Compared model performance using **Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix**.
* Visualized feature importance to understand the factors influencing predictions.

## 📊 Key Results

* **LightGBM** achieved the best overall performance after hyperparameter tuning with an **F1-Score of 0.5577**.
* All three models achieved **ROC-AUC scores above 0.93**, demonstrating excellent classification capability.
* Hyperparameter tuning improved model performance, particularly for **XGBoost**, which showed the largest gain in F1-Score.
* The most influential features were **duration, month, age, and balance**.

## 🔍 Key Insights

* The dataset is significantly **imbalanced**, making **F1-Score** a more suitable evaluation metric than accuracy alone.
* **Duration** was the strongest predictor of subscription, although it represents a potential **data leakage** feature in real-world deployment.
* Customer demographics and financial information, such as **age** and **balance**, also contributed significantly to prediction performance.

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Matplotlib & Seaborn

## 🎯 Conclusion

This project demonstrates the complete machine learning workflow, including **data preprocessing, feature engineering, hyperparameter tuning, model evaluation, and feature importance analysis**. It also highlights the importance of selecting appropriate evaluation metrics for imbalanced datasets and comparing multiple boosting algorithms to identify the most effective model.

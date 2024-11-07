# Machine Learning Model Training and Evaluation

This project trained multiple machine learning models on the Surveillance,Epidemiology, and End Results(SEER) Public-Use breast cancer dataset, performing feature selection, tuning hyperparameters, and evaluating model performance. 

## Project Steps

### Step 1: Data Preprocessing

1. **Data Cleaning**: Inspect and clean the data, handling any missing values or outliers as necessary.
2. **Encoding Categorical Data**: Use one-hot encoding to transform categorical variables into binary indicator variables.
3. **Feature Normalization**: Standardize numerical features to improve model convergence and performance.
4. **Dimensionality Reduction**: PCA was used for dimensionality reduction, capturing maximum variance in data.

### Step 2: Modeling

In this step, we trained six different algorithms on the dataset. We performed feature selection, then trained each model, and evaluated it.

1. **Feature Selection**: 
   - Applied Recursive Feature Elimination (REF) feature selection methods to 90% threshold of PCA components for features ranking and selection.

2. **Algorithms Used**:
   - **K-Nearest Neighbors (KNN)**: A simple instance-based learning algorithm that classifies based on the majority class of the closest neighbors.
   - **Naïve Bayes**: A probabilistic classifier based on Bayes' Theorem with an independence assumption.
   - **C4.5 Decision Tree**: A tree-based algorithm that splits data based on information gain, producing interpretable trees.
   - **Random Forest**: An ensemble of decision trees that reduces overfitting and improves accuracy by combining multiple trees.
   - **Gradient Boosting**: A boosting method that builds sequential trees to correct errors of previous trees, improving overall accuracy.
   - **Neural Network**: A deep learning model with interconnected layers of neurons, capable of learning complex patterns.

### Step 3: Hyperparameter Tuning

We performed hyperparameter tuning on two algorithms, **Nerual Network** and **Random Forest**, to optimize their performance.

1. **Nerual Network**: Used Random Search to tune parameters such as `learning_rate_init`, `hidden_layer_sizes`, `alpha`, `max_iter`, and `activation`. Grid Search explore various combinations of hyperparameters for the Neural Network.
2. **Random Forest**: Used Grid Search to tune parameters such as `n_estimators`, `max_depth`, and `max_features`. Grid Search exhaustively tests all possible combinations in the specified parameter grid.


Each hyperparameter search used cross-validation (5-fold) to get a reliable estimate of performance and selected the best parameters based on cross-validation accuracy.

### Step 4: Results
After tuning, we evaluated each model using metrics such as Accuracy, Precision, Recall, and F1-Score on the test set and combined all results in the following table:


| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| KNN (k=11)           | 0.8911   | 0.89      | 0.89   | 0.88     |
| Naive Bayes          | 0.8648   | 0.85      | 0.86   | 0.85     |
| Decision Tree        | 0.8373   | 0.83      | 0.84   | 0.83     |
| Random Forest (RF)   | 0.8836   | 0.88      | 0.88   | 0.86     |
| RF Optimized         | 0.8861   | 0.88      | 0.89   | 0.87     |
| Gradient Boosting    | 0.8836   | 0.88      | 0.88   | 0.87     |
| Neural Network (NN)  | 0.8861   | 0.88      | 0.89   | 0.87     |
| NN Optimized         | 0.8911   | 0.89      | 0.89   | 0.88     |


We see that KNN and optimized NN achieved highest accuracy (0.8911) as well as precision, recall, and F1-score This suggests that these two models have effectively captured patterns in the data. Naive Bayes and Decision Tree showed slightly lower metrics but still reasonable results. These two models can be suitable for smaller datasets but for our complex dataset, they show limited performance. On the other hand, the ensemble method of the decision tree is able to achieve higher metrics showing that by incorporating multiple decision trees, it aggregates single decision tree results and shows better performance in dealing with complex data. 


Observing classification reports for each model, partly because of the imbalance dataset, all models are better at predicting survived labels than dead labels. Overall, the difference between best performing models and worst ones is small, most models have good performance in all metrics. 

### Step 5: Conclusion


The results indicate that KNN and the optimized Neural Network performed best in terms of overall metrics, achieving high accuracy, precision, recall, and F1-score. Ensemble methods, particularly Random Forest and Gradient Boosting, also performed well, showing strong predictive capabilities with balanced metrics. The optimization of hyperparameters for Neural Network and Random Forest provided slight improvements in performance, confirming the value of hyperparameter tuning in refining model effectiveness. 

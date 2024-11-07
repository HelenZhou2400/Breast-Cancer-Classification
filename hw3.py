import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



path = "C:/Users/hairu/OneDrive/Desktop/PA/hw/hw3/Breast_Cancer_dataset.csv"
df_original = pd.read_csv(path)
df = df_original.copy()

missing_stats = df.isnull().sum()
print("missing values: \n")
print(missing_stats[missing_stats > 0])

###############################################################
#  Step 1 Preprocessing
###############################################################
# -------------- handle missing values --------------

# impute missing numerical values with median, categorical with most frequent 
numeric_features = df.select_dtypes(include=['number']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

if numeric_features:
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

if categorical_features:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True).astype(int)

missing_stats = df.isnull().sum()
print("missing values: \n")
print(missing_stats[missing_stats > 0])
print("finished imputing missing data with median and mode")
# alternatively, can predict and impute missing values using decision tree


# -------------- find outliers --------------
# detect outliers using LOF
lof = LocalOutlierFactor(n_neighbors=20, 
                        contamination='auto',
                        novelty=False)

outlier_labels = lof.fit_predict(df)
df["Outlier"] = outlier_labels
outlier_count = (outlier_labels == -1).sum()
print("Number of outliers using LOF method: ", outlier_count)

outlier_scores = -lof.negative_outlier_factor_
df['LOF_score'] = outlier_scores
df.sort_values(by='LOF_score', ascending=False).head(10)

# visualize outlier
for feature in df.columns[:-1]:  
    plt.scatter(df['LOF_score'], df[feature], alpha=0.5, label=feature)
plt.xlabel('Outlier Score (LOF)')
plt.ylabel('Feature Values')
plt.title('Feature Values vs. Outlier Score')
plt.legend(loc='best', bbox_to_anchor=(1.6, 1))
plt.grid(True)
plt.show()

df = df[df['Outlier'] !=-1].drop(columns=['LOF_score','Outlier'])


# -------------- standardize data--------------
# apply standardization to numerical columns only
y = df['Status_Dead']
X = df.drop('Status_Dead', axis=1)
numericalX = X[numeric_features]
categoricalX = X.drop(numeric_features,axis=1)

numericalX_st = pd.DataFrame(StandardScaler().fit_transform(numericalX),columns=numeric_features)
 
for feature in numericalX_st.columns:  
    plt.scatter(numericalX_st.index, numericalX_st[feature], alpha=0.5, label=feature)
plt.xlabel('Index')
plt.ylabel('Feature Values')
plt.title('Feature Values vs. Index')
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
plt.grid(True)
plt.show()

X_std = categoricalX.reset_index(drop=True).join(numericalX_st.reset_index(drop=True))

# -------------- PCA --------------
# dimensionality reduction using PCA
pca = PCA().fit(X_std)
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# adopt a 85% threshold, reduce 29 features to 9 components
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Optimal Number of PCA Components')
plt.axhline(y=0.90, color='r', linestyle='--')
plt.grid()
plt.show()

###############################################################
#  Step 2 Modeling
###############################################################
# feature ranking is done by examining the PCA variance ratio
threshold = 0.90
num_components = np.argmax(explained_variance >= threshold) 
X_pca = pd.DataFrame(pca.transform(X_std)[:, :num_components],columns=['PC1', 'PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9']) 

# component variance ratio
variance_ratio_table = pd.DataFrame({
    'PCA Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Explained Variance Ratio': np.round(pca.explained_variance_ratio_, 2)
})


# find feature importance for each compoenent
loadings = pd.DataFrame(pca.components_.T, 
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)], 
                        index=X_std.columns)
loadings_abs = loadings.abs()
# rank important features 
ranked_features = loadings_abs.apply(lambda x: x.sort_values(ascending=False).index.tolist(), axis=0)

# output_path = "feature_ranked.csv"  
# ranked_features.to_csv(output_path, index=False)

# feature selection using RFE with a Random Forest, recursively removes least important component
selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)  # Selecting top 5 features
X_selected = selector.fit_transform(X_pca, y)

print("Finished feature ranking and selection")
X_selected.shape    # (3994,5)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42,)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
print("Finished train test set split")

# # Trying to balance data with SMOTE
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# #  Try with undersampler
# undersampler = RandomUnderSampler(random_state=42, sampling_strategy=0.9)
# X_train, y_train = undersampler.fit_resample(X_train, y_train)

# print("Finished balancing data  ")

# -------------- KNN model --------------

def cal_dist(row1, row2):
    dist = 0
    for i in range(len(row1)):
        dist += (row1[i] - row2[i])**2
    return dist**0.5

def knn(X_train, X_test, y_train, k):
    # predict y_pred for X_test
    y_pred = []
    for i in range(len(X_test)):
        distances = []
        # find distance from a test pts to all train pts
        for j in range(len(X_train)):
            distances.append(cal_dist(X_test[i], X_train[j]))
        # get top k closest pts with lowest distances
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[idx] for idx in k_indices]
        # most common label in nearby pts (key)
        y_pred.append(max(set(k_nearest_labels), key=k_nearest_labels.count))
    return np.array(y_pred)

# for verification with sklearn KNeighborsClassifier
def verify_knn(X_train, X_test, y_train, k): 
    y_pred_knn= knn(X_train, X_test, y_train, k)
    print(y_pred_knn[y_pred_knn == 1].shape)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    print(knn_pred[knn_pred == 1].shape)
    return knn_pred[knn_pred == y_pred_knn].shape
# verify_knn(X_train, X_test, y_train, k=5)

accuracies = []
print("Finding optimal K for KNN")
for k in range(1,16):
    y_pred_knn = knn(X_train, X_test, y_train, k)
    accuracy = accuracy_score(y_test, y_pred_knn)
    accuracies.append(accuracy)
    print(f"k = {k}, Accuracy = {accuracy:.2f}")
# find optimal k is 11
top_k = np.argsort(accuracies)[-1]+1
print(f"Optimal K is {top_k} with accuracy {accuracies[top_k-1]:.2f}")  

k = top_k  
y_pred_knn= knn(X_train, X_test, y_train, k)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


# -------------- Naïve Bayes --------------
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


# -------------- C4.5 Decision Tree --------------
dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# -------------- Random Forest --------------
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# -------------- Gradient Boosting --------------
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

# -------------- Neural Network --------------
nn = MLPClassifier(hidden_layer_sizes=(30,), max_iter=100, random_state=42)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn))

###############################################################
#  Step 3  Hyperparameter Tuning
###############################################################
# -------------- random search for NN --------------
param_dist_nn = {
    'learning_rate_init': [0.0001, 0.005, 0.001, 0.01, 0.1],
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'alpha': [0.001, 0.005, 0.01, 0.05, 0.1],  
    'max_iter': [200, 500],
    'activation': ['relu', 'tanh']
}

nn = MLPClassifier(random_state=42)
# cv cross-validation set to 5 folds
random_search_nn = RandomizedSearchCV(estimator=nn, param_distributions=param_dist_nn, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random_search_nn.fit(X_train, y_train)

print("Best Parameters for Neural Network:", random_search_nn.best_params_)
print("Neural Network Best Cross-Validation Accuracy:", random_search_nn.best_score_)

y_pred_nn = random_search_nn.best_estimator_.predict(X_test)
print("Neural Network Test Accuracy:", accuracy_score(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn))

# -------------- grid search for Random Forest --------------
param_grid_rf = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [5, 10, 20, None],
            'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Random Forest Best Cross-Validation Accuracy:", grid_search_rf.best_score_)

y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)
print("Random Forest Test Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


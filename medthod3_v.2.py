import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Read dataset
df = pd.read_csv("C:/Users/aepaw/Downloads/creditcard.csv")

# Features and Target
X = df.drop('Class', axis=1)
y = df['Class']

# Check imbalance
print(df['Class'].value_counts())

# Apply SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print(pd.Series(y_res).value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 3: RFE + PCA
log_reg = LogisticRegression()
rfe = RFE(log_reg, n_features_to_select=15)
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)

# PCA after RFE
pca = PCA(n_components=2)
X_train_pca_rfe = pca.fit_transform(X_train_rfe)
X_test_pca_rfe = pca.transform(X_test_rfe)

# Scree plot for RFE + PCA
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot (RFE + PCA)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Model evaluation function
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Models list
models = [
    ("SVM", SVC(kernel='rbf')),
    ("KNN", KNeighborsClassifier(n_neighbors=3)),
    ("Logistic Regression", LogisticRegression()),
    ("Random Forest", RandomForestClassifier()),
    ("ANN", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)),
    ("Decision Tree", DecisionTreeClassifier())
]

# Evaluate models on RFE + PCA
for name, model in models:
    evaluate_model(name, model, X_train_pca_rfe, X_test_pca_rfe, y_train, y_test)

# Method 4: ANOVA + PCA
anova = SelectKBest(score_func=f_classif, k=15)
X_train_anova = anova.fit_transform(X_train_scaled, y_train)
X_test_anova = anova.transform(X_test_scaled)

pca = PCA(n_components=2)
X_train_pca_anova = pca.fit_transform(X_train_anova)
X_test_pca_anova = pca.transform(X_test_anova)

# Scree plot for ANOVA + PCA
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot (ANOVA + PCA)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Evaluate models on ANOVA + PCA
for name, model in models:
    evaluate_model(name, model, X_train_pca_anova, X_test_pca_anova, y_train, y_test)

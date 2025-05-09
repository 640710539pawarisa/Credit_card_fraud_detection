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
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("C:/Users/aepaw/Downloads/creditcard.csv")
df

# Explore data
df.describe()
print("-------------------------------------------------------------------------------------")

# Check for missing values
df.isnull().sum()
print("-------------------------------------------------------------------------------------")


# Features
X = df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
# Target
y = df['Class']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Split data

# Features
X = df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
# Target
y = df['Class']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#method3
# Import necessary libraries for RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Feature selection using RFE (Recursive Feature Elimination)
log_reg = LogisticRegression()
rfe = RFE(log_reg, n_features_to_select=15)  # Selecting top 15 features
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)

# Check which features were selected
print(f'Selected Features: {rfe.support_}')
print("-------------------------------------------------------------------------------------")

# Apply PCA after RFE with 2 components
pca = PCA(n_components=2)
X_train_pca_rfe = pca.fit_transform(X_train_rfe)
X_test_pca_rfe = pca.transform(X_test_rfe)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by components after RFE: {explained_variance}')
print("-------------------------------------------------------------------------------------")

# Scree plot (already done)
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot (RFE + PCA)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Now use your original models with the PCA-transformed data after RFE

# 1. SVM
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_pca_rfe, y_train)
y_pred = svm_model.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("SVM Results (RFE + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 2. KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_pca_rfe, y_train)
y_pred = knn_model.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("KNN Results (RFE + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 3. Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_pca_rfe, y_train)
y_pred = logistic_model.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Logistic Regression Results (RFE + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 4. Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_pca_rfe, y_train)
y_pred = rf_model.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Random Forest Results (RFE + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 5. ANN
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
clf.fit(X_train_pca_rfe, y_train)
y_pred = clf.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("ANN Results (RFE + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 6. Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train_pca_rfe, y_train)
y_pred = model.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Decision Tree Results (RFE + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix: ')
print(conf_matrix)
print("Classification Report: ")
print(class_report)
print("-------------------------------------------------------------------------------------")


# Import necessary libraries for ANOVA feature selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Feature selection using ANOVA (f_classif)
anova = SelectKBest(score_func=f_classif, k=15)
X_train_anova = anova.fit_transform(X_train_scaled, y_train)
X_test_anova = anova.transform(X_test_scaled)

# Apply PCA after ANOVA with 2 components
pca = PCA(n_components=2)
X_train_pca_anova = pca.fit_transform(X_train_anova)
X_test_pca_anova = pca.transform(X_test_anova)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by components after ANOVA: {explained_variance}')
print("-------------------------------------------------------------------------------------")

# Scree plot
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot (ANOVA + PCA)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Now use your original models with the PCA-transformed data after ANOVA

# 1. SVM
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_pca_anova, y_train)
y_pred = svm_model.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("SVM Results (ANOVA + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 2. KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_pca_anova, y_train)
y_pred = knn_model.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("KNN Results (ANOVA + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 3. Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_pca_anova, y_train)
y_pred = logistic_model.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Logistic Regression Results (ANOVA + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 4. Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_pca_anova, y_train)
y_pred = rf_model.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Random Forest Results (ANOVA + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 5. ANN
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
clf.fit(X_train_pca_anova, y_train)
y_pred = clf.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("ANN Results (ANOVA + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 6. Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train_pca_anova, y_train)
y_pred = model.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Decision Tree Results (ANOVA + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix: ')
print(conf_matrix)
print("Classification Report: ")
print(class_report)
print("-------------------------------------------------------------------------------------")


#smote

# Check for imbalance
print(df['Class'].value_counts())

#smote
# Handle data imbalance using SMOTE

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
sm

# Confirming the new class distribution after SMOTE

print(pd.Series(y_res).value_counts())
print("-------------------------------------------------------------------------------------")

#Split data

# Features
X = df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
# Target
y = df['Class']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Check the size of the split

print("X_res shape:", X_res.shape)

print("X_train size:", X_train.shape) #80%×568,630 (Training samples)
print("X_test size:", X_test.shape)   #20%×568,630 (Testing samples)
print("-------------------------------------------------------------------------------------")

#method3AfterSmote
# Import necessary libraries for RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Feature selection using RFE (Recursive Feature Elimination)
log_reg = LogisticRegression()
rfe = RFE(log_reg, n_features_to_select=15)  # Selecting top 15 features
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)

# Check which features were selected
print(f'Selected Features: {rfe.support_}')
print("-------------------------------------------------------------------------------------")

# Apply PCA after RFE with 2 components
pca = PCA(n_components=2)
X_train_pca_rfe = pca.fit_transform(X_train_rfe)
X_test_pca_rfe = pca.transform(X_test_rfe)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by components after RFE: {explained_variance}')
print("-------------------------------------------------------------------------------------")

# Scree plot (already done)
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot (RFE + PCA)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Now use your original models with the PCA-transformed data after RFE

# 1. SVM
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_pca_rfe, y_train)
y_pred = svm_model.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("SVM Results (RFE + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 2. KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_pca_rfe, y_train)
y_pred = knn_model.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("KNN Results (RFE + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 3. Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_pca_rfe, y_train)
y_pred = logistic_model.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Logistic Regression Results (RFE + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 4. Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_pca_rfe, y_train)
y_pred = rf_model.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Random Forest Results (RFE + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 5. ANN
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
clf.fit(X_train_pca_rfe, y_train)
y_pred = clf.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("ANN Results (RFE + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 6. Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train_pca_rfe, y_train)
y_pred = model.predict(X_test_pca_rfe)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Decision Tree Results (RFE + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix: ')
print(conf_matrix)
print("Classification Report: ")
print(class_report)
print("-------------------------------------------------------------------------------------")


# Import necessary libraries for ANOVA feature selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Feature selection using ANOVA (f_classif)
anova = SelectKBest(score_func=f_classif, k=15)
X_train_anova = anova.fit_transform(X_train_scaled, y_train)
X_test_anova = anova.transform(X_test_scaled)

# Apply PCA after ANOVA with 2 components
pca = PCA(n_components=2)
X_train_pca_anova = pca.fit_transform(X_train_anova)
X_test_pca_anova = pca.transform(X_test_anova)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by components after ANOVA: {explained_variance}')
print("-------------------------------------------------------------------------------------")

# Scree plot
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot (ANOVA + PCA)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Now use your original models with the PCA-transformed data after ANOVA

# 1. SVM
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_pca_anova, y_train)
y_pred = svm_model.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("SVM Results (ANOVA + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 2. KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_pca_anova, y_train)
y_pred = knn_model.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("KNN Results (ANOVA + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 3. Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_pca_anova, y_train)
y_pred = logistic_model.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Logistic Regression Results (ANOVA + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 4. Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_pca_anova, y_train)
y_pred = rf_model.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Random Forest Results (ANOVA + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 5. ANN
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
clf.fit(X_train_pca_anova, y_train)
y_pred = clf.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("ANN Results (ANOVA + PCA):")
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
print("-------------------------------------------------------------------------------------")

# 6. Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train_pca_anova, y_train)
y_pred = model.predict(X_test_pca_anova)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Decision Tree Results (ANOVA + PCA):")
print(f"Accuracy: {accuracy}")
print('Confusion Matrix: ')
print(conf_matrix)
print("Classification Report: ")
print(class_report)
print("-------------------------------------------------------------------------------------")



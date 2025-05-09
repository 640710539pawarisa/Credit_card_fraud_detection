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

# การโหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv("C:/Users/aepaw/Downloads/creditcard.csv")
print(df)
print("-------------------------------------------------------------------------------------")

# การตรวจสอบค่าที่ขาดหาย (Missing values)
print(df.isnull().sum())
print("-------------------------------------------------------------------------------------")

# การแบ่งข้อมูล (Train/Test Split)
# Features
X = df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
# Target
y = df['Class']

# แบ่งข้อมูลออกเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# การปรับขนาดข้อมูล (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# การเลือกคุณสมบัติด้วย RFE (Recursive Feature Elimination)
log_reg = LogisticRegression(max_iter=10000)
rfe = RFE(log_reg, n_features_to_select=15)  # เลือก 15 คุณสมบัติที่สำคัญที่สุด
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)

# ดูว่าฟีเจอร์ไหนถูกเลือกบ้าง
print(f'Selected Features: {rfe.support_}')
print("-------------------------------------------------------------------------------------")

# การทำ PCA (Principal Component Analysis)
# ลดมิติข้อมูลหลังจากการเลือกฟีเจอร์ด้วย RFE ให้เหลือ 2 มิติ
pca = PCA(n_components=2)
X_train_pca_rfe = pca.fit_transform(X_train_rfe)
X_test_pca_rfe = pca.transform(X_test_rfe)

# การแสดงผลสัดส่วนความแปรปรวน (Explained Variance) ของแต่ละ Principal Component
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by components after RFE: {explained_variance}')
print("-------------------------------------------------------------------------------------")

# การสร้าง Scree Plot
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot (RFE + PCA)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()
print("-------------------------------------------------------------------------------------")

# การฝึกโมเดลและประเมินผล (SVM)
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

# การฝึกโมเดลและประเมินผล (KNN)
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

# การฝึกโมเดลและประเมินผล (Logistic Regression)
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

# การฝึกโมเดลและประเมินผล (Random Forest)
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

# การฝึกโมเดลและประเมินผล (ANN)
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

# การฝึกโมเดลและประเมินผล (Decision Tree)
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

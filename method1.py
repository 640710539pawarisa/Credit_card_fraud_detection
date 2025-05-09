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

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv("C:/Users/aepaw/Downloads/creditcard.csv")
print("-------------------------------------------------------------------------------------")

# แสดงสถิติข้อมูลเบื้องต้น เพื่อให้เราเห็นข้อมูลใน dataset ทั้งหมด เช่น ค่าเฉลี่ย ค่าเบี่ยงเบนมาตรฐาน ฯลฯ
print(df.describe())
print("-------------------------------------------------------------------------------------")

# ตรวจสอบค่าว่างในข้อมูล เช่น ข้อมูลที่หายไป หรือ null values
print(df.isnull().sum())
print("-------------------------------------------------------------------------------------")

# แยก Features (X) และ Target (y) ในข้อมูล
# Features = Time, V1-V28, Amount
# Target = Class (Class เป็นตัวแปรที่บอกว่าธุรกรรมนั้นเป็นการทุจริตหรือไม่)
X = df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
y = df['Class']

# แบ่งข้อมูลออกเป็นชุดฝึก (train) และชุดทดสอบ (test) โดยใช้ 80% สำหรับฝึกและ 20% สำหรับทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("-------------------------------------------------------------------------------------")

# 1. **Support Vector Machine (SVM)** - ใช้ Kernel RBF เพื่อจำแนกข้อมูล
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)  # ฝึก SVM กับข้อมูลฝึก
y_pred = svm_model.predict(X_test)  # ทำนายผลลัพธ์จากข้อมูลทดสอบ
print("SVM Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")  # ความแม่นยำ
print('Confusion Matrix:')  # แสดง Confusion Matrix ที่ใช้วัดประสิทธิภาพของโมเดล
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')  # รายงานผลการจำแนกประเภท เช่น Precision, Recall, F1-Score
print(classification_report(y_test, y_pred))
print("-------------------------------------------------------------------------------------")

# 2. **K-Nearest Neighbors (KNN)** - ใช้จำนวน k=3 เพื่อจำแนกข้อมูล
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print("KNN Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print("-------------------------------------------------------------------------------------")

# 3. **Logistic Regression** - ใช้ Logistic Regression เพื่อทำนายผล
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print("Logistic Regression Results:")
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print("-------------------------------------------------------------------------------------")

# 4. **Random Forest** - ใช้ Random Forest Classifier เพื่อทำการทำนาย
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Random Forest Results:")
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print("-------------------------------------------------------------------------------------")

# 5. **Artificial Neural Network (ANN)** - ใช้ MLPClassifier เพื่อจำแนกข้อมูล
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("ANN Results:")
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print("-------------------------------------------------------------------------------------")

# 6. **Decision Tree** - ใช้ DecisionTreeClassifier เพื่อตัดสินใจในการจำแนก
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Decision Tree Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))
print("Classification Report: ")
print(classification_report(y_test, y_pred))
print("-------------------------------------------------------------------------------------")

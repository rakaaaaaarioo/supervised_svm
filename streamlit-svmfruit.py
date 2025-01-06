import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load model Perceptron
model_file = 'svmfruit.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Load scaler
scaler_file = 'scaler_svm.pkl'
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

# Load dataset
file_path = 'fruit.xlsx'
df = pd.read_excel(file_path)
X = df[['diameter', 'weight', 'red', 'green', 'blue']]  # Fitur
y = df['name']  # Label target

# Mapping label ke kelas secara manual (untuk membalik logika)
label_to_class = {'grapefruit': 0, 'orange': 1}  # Pastikan sesuai
class_to_label = {v: k for k, v in label_to_class.items()}  # Membalik mapping

# Fungsi untuk prediksi
def predict_fruit(features):
    features_scaled = scaler.transform([features])
    prediction_class = model.predict(features_scaled)[0]  # Prediksi kelas
    prediction_label = class_to_label[prediction_class]  # Mapping ke label
    return prediction_label, prediction_class

# Konfigurasi Streamlit
st.title("Aplikasi Prediksi Buah Menggunakan SVM")
st.write("Masukkan fitur buah untuk memprediksi jenis buah.")

# Input pengguna
input_features = []
for col in X.columns:
    value = st.number_input(f"Masukkan nilai untuk {col}:", value=0.0)
    input_features.append(value)

# Prediksi
if st.button("Prediksi"):
    label, class_index = predict_fruit(input_features)
    st.success(f"Model memprediksi jenis buah: {label} (Cluster: {class_index})")

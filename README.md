# Tugas UTS: Linear Regression & Logistic Regression
**Mata Kuliah:** Machine Learning  
**Semester:** Ganjil 2026/2027  

---

## 👤 Identitas Mahasiswa
- **Nama**  : Trias Saputra
- **NIM**   : 241011403328
- **Kelas** : Reg B
- **`random_state`** : **28** *(2 digit terakhir NIM: 2410114033**28**)*

---

## 📂 Struktur Repositori
```text
uts-machine-learning-241011403328/
├── README.md            # Dokumentasi lengkap, identitas, dan ringkasan hasil evaluasi
├── tugas_UTS.ipynb      # Jupyter Notebook lengkap yang sudah dieksekusi (Run All)
├── Ads_Data.csv         # Dataset regresi linear (Biaya Iklan vs Penjualan, 40 data)
└── Email_Data.csv       # Dataset klasifikasi logistik (Fitur Link & Kata Promo, 60 data)
```

---

## 📊 Ringkasan Hasil Eksperimen

### 1. Bagian A – Linear Regression (`Ads_Data.csv`)
Model regresi linear dilatih untuk memprediksi volume penjualan bulanan toko online berdasarkan anggaran biaya iklan menggunakan metode **Normal Equation** dan **Gradient Descent** murni berbasis NumPy.

- **Persamaan Garis Regresi**:  
  $$\hat{y} = 28.0950 + 9.0729 \cdot X$$
  - **Intercept ($b = 28.0950$)**: Estimasi penjualan awal sebesar **28.0950 juta Rp** jika toko tidak mengeluarkan biaya promosi sama sekali ($X = 0$).
  - **Koefisien ($m = 9.0729$)**: Setiap kenaikan biaya promosi sebesar **1 juta Rp**, penjualan diprediksi akan bertambah sebesar **9.0729 juta Rp** (*positive return on ad spend*).

- **Tabel Evaluasi Model**:
  | Metrik Evaluasi | Data Training (32 data) | Data Testing (8 data) | Keterangan |
  | :--- | :---: | :---: | :--- |
  | **MAE** | 4.5007 | 4.7281 | Rata-rata selisih absolut penjualan |
  | **MSE** | 37.7311 | 29.7693 | Kuadrat rata-rata galat |
  | **RMSE** | 6.1426 | 5.4561 | Simpangan baku prediksi (juta Rp) |
  | **$R^2$ (Koefisien Determinasi)** | **0.9460 (94.60%)** | **0.9082 (90.82%)** | Sangat baik (*well-fitted*, no overfitting) |

- **Eksperimen Learning Rate pada Gradient Descent (5000 Iterasi)**:
  - $\alpha = 0.001$: **Terlalu Lambat**. Setelah 5000 iterasi, nilai parameter $b \approx 25.8232, m \approx 9.4225$, masih belum mencapai titik minimum optimal.
  - $\alpha = 0.02$: **Optimal / Pas**. Konvergen mulus dan stabil ke $b = 28.0950, m = 9.0729$ (100% identik dengan hasil Normal Equation).
  - $\alpha = 0.1$: **Gagal Konvergen / Divergen**. Langkah update terlalu besar sehingga melompati lembah minimum dan terjadi fenomena *exploding gradients* (`RuntimeWarning: overflow encountered`).

- **Prediksi Nilai Baru & Analisis Ekstrapolasi**:
  - Biaya Iklan **4.5 juta Rp** $\rightarrow$ Penjualan = **68.923 juta Rp** (Interpolasi, reliabel)
  - Biaya Iklan **7.0 juta Rp** $\rightarrow$ Penjualan = **91.605 juta Rp** (Interpolasi, reliabel)
  - Biaya Iklan **12.0 juta Rp** $\rightarrow$ Penjualan = **136.969 juta Rp** (Ekstrapolasi, di luar rentang data historis 0.8–9.9 juta Rp).  
    *Analisis*: Tingkat reliabilitas rendah karena mengasumsikan linearitas tanpa batas; pada kenyataannya berlaku hukum *diminishing returns* dan kejenuhan pasar.

---

### 2. Bagian B – Logistic Regression (`Email_Data.csv`)
Model klasifikasi biner untuk menyaring email spam berdasarkan fitur `JumlahLink` dan `KataPromo` menggunakan **Gradient Descent** dengan **Log Loss** (*Binary Cross-Entropy*).

- **Persamaan Probabilitas Model**:  
  $$P(\text{Spam}) = \sigma(-9.6418 + 0.8467 \cdot \text{JumlahLink} + 0.8027 \cdot \text{KataPromo})$$

- **Persamaan Decision Boundary ($P = 0.5$)**:  
  $$\text{KataPromo} = 12.0118 - 1.0548 \cdot \text{JumlahLink}$$

- **Tabel Evaluasi Confusion Matrix & Metrik pada Data Testing (15 Data, Threshold = 0.5)**:
  | Prediksi \ Aktual | Aktual: Bukan Spam (0) | Aktual: Spam (1) |
  | :--- | :---: | :---: |
  | **Prediksi: Bukan Spam (0)** | **TN = 7** | **FN = 1** |
  | **Prediksi: Spam (1)** | **FP = 1** | **TP = 6** |

  - **Akurasi** : **86.67%** (13 dari 15 data benar)
  - **Precision**: **85.71%**
  - **Recall**   : **85.71%**
  - **F1-Score** : **85.71%**

- **Eksperimen Pemilihan Threshold**:
  | Threshold | TN | FP | FN | TP | Precision | Recall | F1-Score | Akurasi |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **0.3** | 6 | 2 | 1 | 6 | 75.00% | 85.71% | 80.00% | 80.00% |
  | **0.5** | 7 | 1 | 1 | 6 | 85.71% | 85.71% | 85.71% | 86.67% |
  | **0.7** *(Dipilih)* | 8 | **0** | 2 | 5 | **100.00%** | 71.43% | 83.33% | **86.67%** |

  *Alasan Pemilihan Threshold 0.7*: Pada filter spam, kesalahan *False Positive* (email penting yang salah dibuang ke folder spam) memiliki resiko kerugian yang jauh lebih fatal dibanding *False Negative*. Ambang batas **0.7** berhasil menekan nilai **FP hingga 0 (Precision 100%)**.

- **Prediksi Email Baru (dengan Threshold 0.7)**:
  | Email | JumlahLink | KataPromo | Probabilitas $P(\text{Spam})$ | Prediksi Kelas | Status |
  | :---: | :---: | :---: | :---: | :---: | :--- |
  | **E1** | 1 | 2 | 0.0008 (0.08%) | **0** | Bukan Spam |
  | **E2** | 5 | 6 | 0.3561 (35.61%) | **0** | Bukan Spam |
  | **E3** | 9 | 10 | 0.9975 (99.75%) | **1** | **Spam** |
  | **E4** | 2 | 11 | 0.7071 (70.71%) | **1** | **Spam** |

---

### 3. Bagian C – Pemahaman Konsep
1. **Nama "Logistic Regression"**: Termasuk klasifikasi karena tujuannya memetakan input ke kelas diskrit. Disebut regresi karena menghitung kombinasi linear kontinu (*log-odds*), di mana fungsi **sigmoid** bertindak memetakan rentang $(-\infty, \infty)$ menjadi rentang peluang $[0, 1]$.
2. **Normal Equation vs Gradient Descent**: Normal Equation menyelesaikan persamaan secara analitik dalam satu langkah ($\theta = (X^T X)^{-1} X^T y$), sedangkan Gradient Descent melakukan optimasi secara iteratif. Gradient descent wajib dipakai saat jumlah fitur sangat besar ($d > 10.000$) atau data berukuran masif di mana invers matriks $\mathcal{O}(d^3)$ menjadi tidak layak.
3. **Data Testing & Overfitting**: Evaluasi pada data testing wajib dilakukan untuk mengukur kemampuan generalisasi model terhadap data baru yang belum pernah dilihat. Overfitting terjadi ketika model menghafal noise data training sehingga performa pada data testing anjlok.
4. **MSE vs Log Loss**: MSE cocok untuk Linear Regression karena menghasilkan loss konveks pada target kontinu. Pada Logistic Regression, fungsi sigmoid membuat MSE berbentuk non-konveks (banyak local minima); oleh sebab itu digunakan Log Loss yang menjamin kurva loss selalu konveks dengan satu titik minimum global.
5. **Perbedaan Angka Antar Mahasiswa**: Disebabkan oleh parameter `random_state` yang diset berdasarkan 2 digit terakhir NIM. Perbedaan seed menghasilkan pembagian sampel baris data training dan testing yang berbeda pada fungsi `train_test_split`.

---

### 4. Bonus – Perbandingan dengan Scikit-Learn
- **Linear Regression**: Model Normal Equation NumPy dan `sklearn.linear_model.LinearRegression` menghasilkan parameter yang **100% identik** ($b = 28.0950, m = 9.0729, R^2 = 0.9082$) karena keduanya sama-sama menyelesaikan metode OLS secara analitik.
- **Logistic Regression**: 
  - NumPy GD: $b = -9.6418, m = [0.8467, 0.8027]$
  - Sklearn Default ($C=1.0, L_2$ penalty): $b = -8.8968, m = [0.7683, 0.7457]$
  - Sklearn Unregularized (`penalty=None`): $b = -10.1362, m = [0.8951, 0.8400]$
  - *Penyebab Perbedaan*: `scikit-learn` secara default menerapkan regularisasi $L_2$ dengan penalti bobot untuk mencegah overfitting. Ketika regularisasi dinonaktifkan, hasilnya mendekati implementasi manual kita. Keduanya menghasilkan **akurasi testing yang sama persis (86.67%)**.

---

## 🛠️ Persyaratan Lingkungan (Dependencies)
Pastikan lingkungan Python Anda memiliki pustaka berikut:
```bash
pip install numpy pandas matplotlib scikit-learn
```

Untuk menjalankan notebook:
```bash
jupyter notebook tugas_UTS.ipynb
```
*(Catatan: Seluruh sel output dan grafik visualisasi sudah dijalankan dan tersimpan di dalam notebook sehingga dapat langsung dilihat saat dibuka di GitHub).*

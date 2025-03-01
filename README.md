# Customer-Segmentation-using-Unsupervised-Machine-Learning-in-Python

Di era sekarang, perusahaan bekerja keras untuk membuat pelanggannya senang. Mereka meluncurkan teknologi dan layanan baru agar pelanggan dapat lebih banyak menggunakan produk mereka. Mereka mencoba untuk berhubungan dengan setiap pelanggan mereka sehingga mereka dapat menyediakan barang yang sesuai. Namun dalam praktiknya, sangat sulit dan tidak realistis untuk tetap berhubungan dengan semua orang. Jadi, di sinilah penggunaan Segmentasi Pelanggan.

Segmentasi Pelanggan berarti segmentasi pelanggan berdasarkan kesamaan karakteristik, perilaku, dan kebutuhan mereka. Hal ini pada akhirnya akan membantu perusahaan dalam banyak hal. Misalnya, mereka dapat meluncurkan produk atau meningkatkan fitur yang sesuai. Mereka juga dapat menargetkan sektor tertentu sesuai perilaku mereka. Semua ini mengarah pada peningkatan nilai pasar perusahaan secara keseluruhan.

# Unique Values

![image](https://github.com/user-attachments/assets/58dd9c7f-9b2c-4bef-bd7f-202342a3af13)

Di sini kita dapat mengamati bahwa ada kolom yang berisi nilai tunggal di seluruh kolom, sehingga tidak relevan dalam pengembangan model.

# Data Visualization and Analysis
Untuk mendapatkan grafik hitungan untuk kolom tipe data – objek, rujuk kode di bawah ini.
![image](https://github.com/user-attachments/assets/e6a0ecf4-7a7c-456f-ba3a-1374505d4e1f)

Mari kita periksa value_counts dari Marital_Status dari data.

![image](https://github.com/user-attachments/assets/9eaaa76e-2513-41e2-a213-003f179a6974)

# Segmentation

![image](https://github.com/user-attachments/assets/4933242b-33e0-4d0d-8259-a9776c981c97)

Tentu saja ada beberapa cluster yang terlihat jelas dari representasi 2-D data yang diberikan. Mari kita gunakan algoritma KMeans untuk menemukan cluster tersebut di bidang dimensi tinggi itu sendiri.

KMeans Clustering  juga dapat digunakan untuk mengelompokkan titik-titik berbeda pada suatu bidang.
```sh
error = []
for n_clusters in range(1, 21):
    model = KMeans(init='k-means++',
                   n_clusters=n_clusters,
                   max_iter=500,
                   random_state=22)
    model.fit(df)
    error.append(model.inertia_)
```
Di sini inersia tidak lain adalah jumlah kuadrat jarak dalam gugus.

```sh
plt.figure(figsize=(10, 5))
sb.lineplot(x=range(1, 21), y=error)
sb.scatterplot(x=range(1, 21), y=error)
plt.show()
```

**Output:**

![image](https://github.com/user-attachments/assets/87a64f03-96ee-4608-b1bc-ea34b3025d52)

Di sini dengan menggunakan metode siku kita dapat mengatakan bahwa k = 6 adalah jumlah klaster optimal yang harus dibuat karena setelah k = 6 nilai inersia tidak menurun drastis.

```sh
# create clustering model with optimal k=5
model = KMeans(init='k-means++',
               n_clusters=5,
               max_iter=500,
               random_state=22)
segments = model.fit_predict(df)
```

Scatterplot akan digunakan untuk melihat keenam klaster yang dibentuk oleh  KMean Clustering.

```sh
plt.figure(figsize=(7, 7))
# Create a DataFrame with the tsne_data and segments
df_tsne = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'segment': segments})
# Use the DataFrame in the scatterplot function
sb.scatterplot(x='x', y='y', hue='segment', data=df_tsne)
plt.show()
```


![image](https://github.com/user-attachments/assets/b1e8b773-3f72-4f19-9d1c-839f05147e70)










# IMDB Sentiment Analysis
Bu proje, IMDB film yorumlarını **pozitif** veya **negatif** olarak sınıflandırmak için farklı makine öğrenmesi algoritmalarını karşılaştırmalı olarak uygulamaktadır.  

## Kullanılan Teknolojiler
- Python 3.x  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn (Logistic Regression, Random Forest, SVC, Multinomial Naive Bayes)  

## Veri Ön İşleme
- Yorumlar temizlenerek sadece harfler bırakılmıştır.  
- Küçük harfe dönüştürme ve noktalama işaretlerini kaldırma işlemleri yapılmıştır.  
- `TfidfVectorizer` ile kelimeler sayısallaştırılmıştır.  

## Kullanılan Modeller
Projede 4 farklı makine öğrenmesi algoritması test edilmiştir:  
1. **Logistic Regression** 
2. **Random Forest Classifier**  
3. **Linear Support Vector Classifier (SVC)**  
4. **Multinomial Naive Bayes**  

Her model için:  
- **Accuracy, Precision, Recall, F1 Score** metrikleri hesaplanmıştır.  
- **Confusion Matrix** görselleştirilmiştir.  

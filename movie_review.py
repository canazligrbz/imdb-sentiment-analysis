import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Yeni fonksiyon: Model performansını değerlendirir ve görselleştirir
def evaluate_and_visualize_model(model_name, y_true, y_pred, metrics_df):
    #Bir modelin metriklerini hesaplar, sonuçları yazdırır ve görselleştirir.
    
    # Metrikleri hesapla
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Sonuçları yazdır
    print(f"{model_name} Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix'i görselleştir
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    # Metrikleri DataFrame'e ekle ve döndür
    return pd.concat([metrics_df, pd.DataFrame([{"Model": model_name, "Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1}])], ignore_index=True)

def main():
    # Veri setini yükle
    data = pd.read_csv('IMDB Dataset.csv')

    # Veriyi temizle ve etiketleri dönüştür
    data['review'] = data['review'].apply(clean_text)
    data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == "positive" else 0)

    # Veriyi Tfidf ile sayısallaştır
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = vectorizer.fit_transform(data['review'])
    y = data['sentiment']
    
    # Eğitim ve test verilerini ayır
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Modelleri ve parametre ızgaralarını tanımla
    models = {
        "Logistic Regression": (
            Pipeline([('model', LogisticRegression(solver="liblinear", random_state=42))]),
            {"model__C": np.logspace(-3, 3, 7), "model__penalty": ["l1", "l2"]}
        ),
        "Random Forest": (
            Pipeline([('model', RandomForestClassifier(random_state=42))]),
            {"model__n_estimators": [100, 300], "model__max_features": [3, 5, "sqrt"],
             "model__min_samples_split": [2, 5], "model__min_samples_leaf": [1, 3], "model__criterion": ["gini"]}
        ),
        "SVC (Linear)": (
            Pipeline([('model', LinearSVC(random_state=42, max_iter=2000))]),
            {"model__C": [0.1, 1, 10]}
        ),
        "Multinomial NB": (
            Pipeline([('model', MultinomialNB())]),
            {"model__alpha": [0.01, 0.1, 1.0]}
        ),
    }

    best_models = {}
    metrics_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

    # Modelleri eğit ve değerlendir
    for model_name, (pipeline, param_grid) in models.items():
        print(f"\n--- {model_name} için RandomizedSearchCV başlatılıyor ---")

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=10,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        y_pred = random_search.predict(X_test)

        # Yeni fonksiyonu çağır
        metrics_df = evaluate_and_visualize_model(model_name, y_test, y_pred, metrics_df)
        
        best_models[model_name] = {
            "best_estimator": random_search.best_estimator_,
            "best_params": random_search.best_params_,
            "test_accuracy": accuracy_score(y_test, y_pred)
        }

    # Sonuçları yazdır
    best_model_name = max(best_models, key=lambda k: best_models[k]['test_accuracy'])
    print(f"\nEn iyi model: {best_model_name} - Test Accuracy: {best_models[best_model_name]['test_accuracy']:.4f}")
    print("En iyi modelin parametreleri:", best_models[best_model_name]['best_params'])
    
    # Tüm metrikleri gösteren son tabloyu yazdır
    metrics_df = metrics_df.sort_values(by="Accuracy", ascending=False)
    print("\n\nKarşılaştırmalı Metrik Tablosu:")
    print(metrics_df)

    # Sonuçları görselleştir
    plt.figure(figsize=(10, 6))
    metrics_plot = metrics_df.set_index("Model")
    metrics_plot.plot(kind="bar", figsize=(12, 6), colormap="viridis")
    plt.ylabel("Skor")
    plt.title("Modellerin Accuracy, Precision, Recall ve F1 Skorları Karşılaştırması")
    plt.ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    main()
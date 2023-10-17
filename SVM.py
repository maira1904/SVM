from sklearn.datasets import make_classification
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Создайте имитационные данные для классификации
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Инициализируйте и обучите SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y)

# Создайте имитационные данные для классификации текстовых документов
documents = ["Это текст первого документа.", "А это второй документ.", "Еще один текст для третьего документа."]
labels = [0, 1, 0]

# Создайте векторизатор TF-IDF для преобразования текстов в числовые признаки
tfidf_vectorizer = TfidfVectorizer()

# Преобразуйте тексты в матрицу TF-IDF признаков
X_text = tfidf_vectorizer.fit_transform(documents)

# Инициализируйте и обучите SVMT (Support Vector Machine for Text)
svmt_classifier = SVC(kernel='linear')
svmt_classifier.fit(X_text, labels)

# Инициализируйте и обучите TSVM (Transductive Support Vector Machine)
tsvm_classifier = NuSVC(kernel='linear')
tsvm_classifier.fit(X, y)

# Делайте предсказания для SVM
y_pred_svm = svm_classifier.predict(X)

# Делайте предсказания для SVMT
X_text_test = tfidf_vectorizer.transform(["Это новый текст для теста."])
y_pred_svmt = svmt_classifier.predict(X_text_test)

# Делайте предсказания для TSVM
y_pred_tsvm = tsvm_classifier.predict(X)

# Оцените точность моделей
accuracy_svm = accuracy_score(y, y_pred_svm)
accuracy_svmt = y_pred_svmt[0]
accuracy_tsvm = accuracy_score(y, y_pred_tsvm)

print("Точность модели SVM:", accuracy_svm)
print("Точность модели SVMT:", accuracy_svmt)
print("Точность модели TSVM:", accuracy_tsvm)

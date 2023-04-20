import pandas as pd
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Wczytanie danych (w tym przypadku używam zbioru danych Wine Quality z internetu)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# Testowanie hipotez statystycznych: sprawdzenie, czy istnieje związek pomiędzy jakością wina a zawartością alkoholu
# H0: współczynnik kierunkowy (slope) równy 0 (brak związku)
# H1: współczynnik kierunkowy różny od 0 (istnieje związek)
x_alcohol = data['alcohol']
y_quality = data['quality']
slope, intercept, r_value, p_value, std_err = stats.linregress(x_alcohol, y_quality)

print(f"Współczynnik kierunkowy: {slope}")
print(f"Wartość p: {p_value}")

if p_value < 0.05:
    print("Odrzucamy hipotezę zerową, istnieje związek między jakością wina a zawartością alkoholu.")
else:
    print("Nie ma wystarczających dowodów na odrzucenie hipotezy zerowej.")

# Analiza regresji wielorakiej
X = data.drop('quality', axis=1)
X = sm.add_constant(X)  # Dodanie stałej do modelu
y = data['quality']

# Budowa modelu regresji wielorakiej
model = sm.OLS(y, X).fit()

# Ocena modelu regresji wielorakiej
print(model.summary())

# Analiza reszt
residuals = model.resid
fitted_values = model.fittedvalues

# Wykres reszt
plt.scatter(fitted_values, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

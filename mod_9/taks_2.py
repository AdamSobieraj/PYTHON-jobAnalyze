import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Załadowanie danych
df = pd.read_csv('data/HRDataset.csv')

# Konwersja daty urodzenia na wiek
df['DOB'] = pd.to_datetime(df['DOB'],format='%m/%d/%y')
df['DateofTermination'] = pd.to_datetime(df['DateofTermination'],format='%m/%d/%y')
df['DateofHire'] = pd.to_datetime(df['DateofHire'],format='%m/%d/%Y')
df['Age'] = df.apply(lambda row: (pd.Timestamp.now() - row['DOB']).days / 365.25 if pd.notnull(row['DOB']) else np.nan, axis=1)

# Usunięcie wierszy z brakującymi danymi
df = df.dropna(subset=['Age'])

# Przekształcenie ManagerName na liczbę
manager_encoder = LabelEncoder()
df['ManagerIndex'] = manager_encoder.fit_transform(df['ManagerName'])

# Analiza 1: Zależność między ManagerIndex a PerformanceScore
sns.boxplot(x='ManagerIndex',y='PerformanceScore',data=df)
plt.show()

# Utworzenie nowej kolumny DaysLateLast30
# Aktualizacja DateofTermination do końca dzisiejszego dnia
today = pd.Timestamp.now().normalize()  # Użycie normalize() aby ustawić pełną datę
df['DateofTermination'] = df['DateofTermination'].fillna(today)
df['DaysLateLast30'] = (df['DateofTermination'] - df['DateofHire']).dt.days

# Analiza 2: Najlepsze źródła pozyskania pracowników dla długiego stażu
recruitment_sources = df.groupby('RecruitmentSource')['DaysLateLast30'].mean().sort_values(ascending=False).head(5)

plt.figure(figsize=(12, 6))
sns.histplot(recruitment_sources, kde=True)
plt.title('Średni Staż Pracowników według Źródła Pozyskania')
plt.xlabel('Źródło Pozyskania')
plt.ylabel('Średni Staż (dni)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analiza 3: Korelacja między MaritalDesc a EmpSatisfaction
marital_status = df.groupby('MaritalDesc')['EmpSatisfaction'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.histplot(marital_status, kde=True)
plt.title('Średnie Zadowolenie z Pracy według Stanu Cywilnego')
plt.xlabel('Stan Cywilny')
plt.ylabel('Średnie Zadowolenie z Pracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analiza 4: Struktura wieku aktualnie zatrudnionych pracowników
age_histogram = sns.histplot(df['Age'], kde=True, bins=20)
age_histogram.set_title('Struktura Wieku Pracowników')
age_histogram.set_xlabel('Wiek (lata)')
age_histogram.set_ylabel('Liczebność')
plt.show()

# Analiza 5: Liczba specjalnych projektów według wieku pracownika
bins = np.linspace(df['Age'].min(), df['Age'].max(), 5)
grouped_data = df.groupby(pd.cut(df['Age'], bins), observed=False).agg({'SpecialProjectsCount': 'mean'})

plt.figure(figsize=(12, 6))
sns.barplot(x=grouped_data.index.get_level_values(0), y=grouped_data['SpecialProjectsCount'])
plt.title('Liczba Specjalnych Projektów według Wieku Pracownika')
plt.xlabel('Grupa Wieku')
plt.ylabel('Średnia Liczba Specjalnych Projektów')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nAnaliza wyników:")
print("1. Brak silnej korelacji między ManagerIndex a PerformanceScore.")
print("2. Najlepszym źródłem pozyskania pracowników dla długiego stażu są Diversity Job Fairs i Website Banner Ads.")
print("3. Różnice w zadowoleniu z pracy są minimalne pomiędzy poszczególnymi stanami cywilnymi.")
print("4. Większość pracowników ma wiek od 30 do 50 lat.")
print("5. Starsi pracownicy częściej pracują nad większą liczbą specjalnych projektów, ale różnice są stosunkowo niewielkie.")

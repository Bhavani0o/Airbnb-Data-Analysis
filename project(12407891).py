# =========================================
# 📌 AIRBNB DATA ANALYSIS PROJECT (FINAL MASTER)
# =========================================

# ---------- 1) IMPORT LIBRARIES ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)


# ---------- 2) LOAD DATA ----------
df = pd.read_csv(r"C:\Users\ACER\Downloads\Airbnb_Open_Data.csv.zip", low_memory=False)

print("Initial Shape:", df.shape)

# =========================================
# 📌 PROJECT OVERVIEW
# =========================================
print("\n================ PROJECT OVERVIEW ================\n")

print("Dataset Name: Airbnb Open Data")
print("Goal: Analyze Airbnb listings and pricing patterns")
print("Target Variable: Price")

print("\nRows & Columns:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# =========================================
# 📊 4. BASIC EDA (EXPLORATORY DATA ANALYSIS)
# =========================================

print("\n================ BASIC EDA ================\n")

# ---------- 1. DATA OVERVIEW ----------
print("First 5 rows:\n", df.head())

print("\nLast 5 rows:\n", df.tail())

print("\nRandom Sample:\n", df.sample(5))


# ---------- 2. UNIQUE VALUES ----------
print("\nUnique Values Per Column:\n")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")


# ---------- 3. TOP CATEGORIES ----------
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    print(f"\nTop categories in {col}:\n", df[col].value_counts().head(5))

# =========================================
# 📊 KPI SUMMARY (SAFE VERSION)
# =========================================

print("\n================ KPI SUMMARY ================\n")

if pd.api.types.is_numeric_dtype(df['price']):

    print("Total Listings:", len(df))
    print("Average Price:", round(df['price'].mean(), 2))
    print("Median Price:", df['price'].median())
    print("Max Price:", df['price'].max())
    print("Min Price:", df['price'].min())

else:
    print("❌ Price column not numeric")

# Other KPIs
if 'room_type' in df.columns:
    print("Most Common Room Type:", df['room_type'].mode()[0])

if 'availability_365' in df.columns:
    print("Avg Availability:", round(df['availability_365'].mean(), 2))

if 'number_of_reviews' in df.columns:
    print("Avg Reviews:", round(df['number_of_reviews'].mean(), 2))

print("\n============================================\n")

# ---------- CLEANING ----------
df = df.drop_duplicates()
df = df.replace(['?', 'NA', 'null'], np.nan)

# Price cleaning
df['price'] = df['price'].replace(r'[\$,]', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Numeric conversion
for col in ['minimum_nights', 'number_of_reviews', 'reviews_per_month']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Date conversion
if 'last_review' in df.columns:
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

# Fill missing values
num_cols = df.select_dtypes(include=['float64','int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=['object','string']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    df[col] = df[col].astype(str).str.lower().str.strip()

# Outlier removal (IQR)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)].copy()

# Drop useless columns
for col in ['id', 'host_id', 'license']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

df.columns = df.columns.str.lower().str.replace(" ", "_")

print("After Cleaning Shape:", df.shape)


# =========================================
# 📊 4. BASIC VISUALS
# =========================================

plt.figure()
sns.histplot(df['price'], bins=40, kde=True)
plt.title("Price Distribution")
plt.show()

if 'room_type' in df.columns:
    plt.figure()
    df['room_type'].value_counts().plot(kind='bar')
    plt.title("Room Type Distribution")
    plt.xticks(rotation=360)
    plt.show()

    plt.figure()
    sns.boxplot(x='room_type', y='price', data=df)
    plt.title("Price by Room Type")
    plt.xticks(rotation=90)
    plt.show()

if 'neighbourhood_group' in df.columns:
    top_expensive = df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False).head(10)

    plt.figure()
    top_expensive.plot(kind='bar')
    plt.title("Top 10 Expensive Locations")
    plt.xticks(rotation=45)
    plt.show()


# =========================================
# 📊 5. SCATTER PLOTS
# =========================================

for col in ['number_of_reviews', 'minimum_nights', 'availability_365']:
    if col in df.columns:
        plt.figure()
        sns.scatterplot(x=col, y='price', data=df, alpha=0.5)
        plt.title(f"Price vs {col}")
        plt.show()



# =========================================
# 📈 6. TIME SERIES
# =========================================

if 'last_review' in df.columns:

    # Ensure datetime conversion AGAIN (important fix)
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    # Drop rows where conversion failed
    df_time = df.dropna(subset=['last_review']).copy()

    if not df_time.empty:

        df_time['year_month'] = df_time['last_review'].dt.to_period('M')

        # Price trend
        monthly_price = df_time.groupby('year_month')['price'].mean()

        plt.figure()
        monthly_price.plot(marker='o')
        plt.title("Monthly Price Trend")
        plt.xticks(rotation=45)
        plt.show()

        # Review trend
        if 'number_of_reviews' in df_time.columns:
            review_trend = df_time.groupby('year_month')['number_of_reviews'].sum()

            plt.figure()
            review_trend.plot(marker='o')
            plt.title("Review Trend")
            plt.xticks(rotation=45)
            plt.show()

    else:
        print("No valid datetime data available for time series.")


# =========================================
# 🚨 OUTLIERS
# =========================================

plt.figure()
sns.boxplot(x=df['price'])
plt.title("Price Outliers")
plt.show()

z_scores = np.abs(stats.zscore(df['price']))
print("Z-score outliers:", (z_scores > 3).sum())


# =========================================
# 📊 7. ADVANCED VISUALS
# =========================================

if 'room_type' in df.columns:
    plt.figure()
    df['room_type'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title("Room Type Share")
    plt.ylabel("")
    plt.show()

plt.figure()
sns.kdeplot(df['price'], fill=True)
plt.title("Price Density")
plt.show()

plt.figure()
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(x='room_type', y='price', data=df, color='lightgray')
sns.stripplot(x='room_type', y='price', data=df, jitter=True, alpha=0.4)

plt.title("Price Distribution (Optimized Visualization)")
plt.show()

#####################################
if 'neighbourhood_group' in df.columns and 'room_type' in df.columns:

    stacked_data = pd.crosstab(df['neighbourhood_group'], df['room_type'])

    stacked_data.plot(kind='bar', stacked=True, figsize=(10,6))

    plt.title("Room Type Distribution Across Locations")
    plt.xlabel("Location")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Room Type")
    plt.show()


# =========================================
# 📊 8. STATISTICAL TESTS
# =========================================

if 'room_type' in df.columns:
    groups = df.groupby('room_type')['price'].apply(list)

    if len(groups) >= 2:
        group1 = groups.iloc[0]
        group2 = groups.iloc[1]

        if len(group1) > 1 and len(group2) > 1:
            t_stat, p_val = stats.ttest_ind(group1, group2)
            print("\nT-Test:", t_stat, p_val)

if 'number_of_reviews' in df.columns:
    if len(df) > 2:
        corr, p_val = stats.pearsonr(df['price'], df['number_of_reviews'])
        print("\nCorrelation:", corr)
        
######################################
if 'neighbourhood_group' in df.columns:

    # Bar data (count)
    count_data = df['neighbourhood_group'].value_counts()

    # Line data (avg price)
    price_data = df.groupby('neighbourhood_group')['price'].mean()

    fig, ax1 = plt.subplots(figsize=(10,6))

    # BAR CHART
    count_data.plot(kind='bar', ax=ax1, alpha=0.7)
    ax1.set_ylabel("Number of Listings")
    ax1.set_xlabel("Location")

    # LINE CHART
    ax2 = ax1.twinx()
    price_data.plot(kind='line', color='red', marker='o', ax=ax2)
    ax2.set_ylabel("Average Price")

    plt.title("Listings Count vs Average Price")
    plt.xticks(rotation=45)
    plt.show()

# =========================================
# 🤖 9. LINEAR REGRESSION
# =========================================

df_ml = df.select_dtypes(include=['float64','int64']).dropna()

X = df_ml.drop(columns=['price'])
y = df_ml['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMODEL PERFORMANCE")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Residual plot
plt.figure()
sns.scatterplot(x=y_pred, y=(y_test - y_pred))
plt.axhline(0, color='red')
plt.title("Residual Plot")
plt.show()


# =========================================
# 📌 FINAL INSIGHTS
# =========================================

print("\nKEY INSIGHTS:")
print("✔ Price depends strongly on room type")
print("✔ Entire homes are more expensive")
print("✔ Weak correlation between reviews and price")
print("✔ Data cleaned and outliers handled")
print("✔ Location and availability affect pricing")

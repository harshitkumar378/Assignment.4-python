import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Task 1: Data Acquisition and Loading
# ---------------------------------------------------------

# Create sample weather dataset
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2023-12-31")

df = pd.DataFrame({
    "date": dates,
    "temperature": np.random.normal(30, 5, len(dates)),
    "rainfall": np.random.choice([0, 2, 5, 10, 20], len(dates)),
    "humidity": np.random.randint(40, 95, len(dates))
})

# Introduce some NaN values to simulate real data
df.loc[10:15, "temperature"] = np.nan
df.loc[100, "rainfall"] = np.nan

print("Initial Data:")
print(df.head(), "\n")


# ---------------------------------------------------------
# Task 2: Data Cleaning & Processing
# ---------------------------------------------------------

# Handle missing values
df["temperature"].fillna(df["temperature"].mean(), inplace=True)
df["rainfall"].fillna(0, inplace=True)

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

# Filter relevant columns
weather = df[["date", "temperature", "rainfall", "humidity"]]

print("Cleaned Data:")
print(weather.head(), "\n")


# ---------------------------------------------------------
# Task 3: Statistical Analysis with NumPy
# ---------------------------------------------------------

daily_mean = np.mean(weather["temperature"])
monthly_stats = weather.groupby(weather["date"].dt.month).agg({
    "temperature": ["mean", "min", "max", "std"],
    "rainfall": "sum",
    "humidity": "mean"
})

yearly_stats = {
    "temp_mean": np.mean(weather["temperature"]),
    "temp_max": np.max(weather["temperature"]),
    "temp_min": np.min(weather["temperature"]),
    "temp_std": np.std(weather["temperature"])
}

print("\nMonthly Statistics:\n", monthly_stats)
print("\nYearly Statistics:\n", yearly_stats)


# ---------------------------------------------------------
# Task 4: Visualization with Matplotlib
# ---------------------------------------------------------

# Line chart: Daily temperature
plt.figure(figsize=(10, 5))
plt.plot(weather["date"], weather["temperature"])
plt.title("Daily Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.savefig("daily_temperature.png")
plt.close()

# Bar chart: Monthly rainfall totals
monthly_rain = weather.groupby(weather["date"].dt.month)["rainfall"].sum()

plt.figure(figsize=(8, 5))
plt.bar(monthly_rain.index, monthly_rain.values)
plt.title("Monthly Rainfall Totals")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.savefig("monthly_rainfall.png")
plt.close()

# Scatter plot: Humidity vs Temperature
plt.figure(figsize=(8, 5))
plt.scatter(weather["humidity"], weather["temperature"])
plt.title("Humidity vs Temperature")
plt.xlabel("Humidity (%)")
plt.ylabel("Temperature (°C)")
plt.savefig("humidity_vs_temperature.png")
plt.close()

# Combined figure
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(weather["date"], weather["temperature"])
ax[0].set_title("Daily Temperature Trend")

ax[1].scatter(weather["humidity"], weather["temperature"])
ax[1].set_title("Humidity vs Temperature")

plt.tight_layout()
plt.savefig("combined_plot.png")
plt.close()


# ---------------------------------------------------------
# Task 5: Grouping and Aggregation
# ---------------------------------------------------------

weather["month"] = weather["date"].dt.month

monthly_summary = weather.groupby("month").agg({
    "temperature": ["mean", "max", "min"],
    "rainfall": "sum",
    "humidity": "mean"
})

print("\nMonthly Summary:\n", monthly_summary)


# ---------------------------------------------------------
# Task 6: Export and Storytelling
# ---------------------------------------------------------

# Export cleaned dataset
weather.to_csv("cleaned_weather_data.csv", index=False)

# Generate a report summary
with open("summary_report.txt", "w") as f:
    f.write("Weather Data Analysis Report\n")
    f.write("--------------------------------------\n\n")
    f.write(f"Yearly Temperature Mean: {yearly_stats['temp_mean']:.2f}\n")
    f.write(f"Yearly Temp Max: {yearly_stats['temp_max']:.2f}\n")
    f.write(f"Yearly Temp Min: {yearly_stats['temp_min']:.2f}\n")
    f.write(f"Yearly Temp Std: {yearly_stats['temp_std']:.2f}\n")
    f.write("\n\nMonthly Rainfall Summary:\n")
    f.write(str(monthly_rain))

print("\nAll files exported successfully!")

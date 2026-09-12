from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data/mess_crowd_dataset.csv")

df.dropna(inplace=True)

day_encoder = LabelEncoder()
meal_encoder = LabelEncoder()
df["Day"] = day_encoder.fit_transform(df["Day"])
df["Meal"] = meal_encoder.fit_transform(df["Meal"])
df["Time"] = (
    pd.to_datetime(df["Time"], format="%H:%M").dt.hour * 60
    + pd.to_datetime(df["Time"], format="%H:%M").dt.minute
)

X = df.drop(columns=["Crowd"])
y = df["Crowd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier( n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.score(X_test, y_test))

day = input("Enter day of the week (e.g., Monday): ")
meal = input("Enter meal type (e.g., Lunch, Dinner): ")
time = input("Enter time in HH:MM format (e.g., 12:30): ")

day = day_encoder.transform([day])[0]
meal = meal_encoder.transform([meal])[0]
time = pd.to_datetime(time, format="%H:%M")
time = time.hour * 60 + time.minute

input_data = pd.DataFrame({
    "Day": [day],
    "Meal": [meal],
    "Time": [time]
})

prediction = model.predict(input_data)

crowd_map = {
    0 : "No Crowd",
    1 : "Moderate Crowd",
    2 : "Heavy Crowd"
}

print(f"\nPredicted Crowd Level: {crowd_map[prediction[0]]}")
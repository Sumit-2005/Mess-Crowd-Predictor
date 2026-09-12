from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("data/mess_crowd_dataset.csv")

df.dropna(inplace=True)

encoder = LabelEncoder()
df["Day"] = encoder.fit_transform(df["Day"])
df["Meal"] = encoder.fit_transform(df["Meal"])
df["Time"] = (
    pd.to_datetime(df["Time"], format="%H:%M").dt.hour * 60
    + pd.to_datetime(df["Time"], format="%H:%M").dt.minute
)

X = df.drop(columns=["Meal", "Crowd"])
y = df["Crowd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.score(X_test, y_test))

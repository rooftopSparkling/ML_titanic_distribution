from flask import Flask, request, render_template
import joblib
import pandas as pd
from preprocess import TitanicPreprocessor

app = Flask(__name__)
model = joblib.load("titanic_pipeline_grid_best.joblib")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        data = {
            "Pclass": int(request.form["Pclass"]),
            "Name": request.form["Name"],
            "Sex": request.form["Sex"],
            "Age": float(request.form["Age"]) if request.form["Age"] else None,
            "SibSp": int(request.form["SibSp"]),
            "Parch": int(request.form["Parch"]),
            "Ticket": request.form["Ticket"],
            "Fare": float(request.form["Fare"]) if request.form["Fare"] else None,
            "Cabin": request.form["Cabin"] if request.form["Cabin"] else None,
            "Embarked": request.form["Embarked"] if request.form["Embarked"] else None,
        }

        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0, 1]

        prediction = pred
        probability = proba

    return render_template("web.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)

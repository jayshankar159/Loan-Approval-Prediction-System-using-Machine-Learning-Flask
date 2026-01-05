from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("loan_pipeline.pkl")

def safe_float(val):
    return float(val) if val.strip() != "" else 0.0

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        data = pd.DataFrame([{
            'Gender': request.form['Gender'],
            'Married': request.form['Married'],
            'Dependents': request.form['Dependents'],
            'Education': request.form['Education'],
            'Self_Employed': request.form['Self_Employed'],
            'ApplicantIncome': safe_float(request.form['ApplicantIncome']),
            'CoapplicantIncome': safe_float(request.form['CoapplicantIncome']),
            'LoanAmount': safe_float(request.form['LoanAmount']),
            'Loan_Amount_Term': safe_float(request.form['Loan_Amount_Term']),
            'Credit_History': safe_float(request.form['Credit_History']),
            'Property_Area': request.form['Property_Area']
        }])

        prediction = model.predict(data)

        result = "✅ Loan Approved" if prediction[0] == 1 else "❌ Loan Rejected"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

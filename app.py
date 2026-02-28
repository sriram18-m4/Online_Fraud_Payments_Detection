from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
import math

app = Flask(__name__)

model = xgb.XGBClassifier()
model.load_model("xgb_fraud_detection_model.json")

TYPE_MAPPING = {
    'CASH_IN': 0,
    'CASH_OUT': 1,
    'DEBIT': 2,
    'PAYMENT': 3,
    'TRANSFER': 4
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        step = int(request.form['step'])
        txn_type = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        amount = math.log(amount) if amount > 0 else 0
        txn_type = TYPE_MAPPING[txn_type]

        input_df = pd.DataFrame([[
            step,
            txn_type,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest
        ]], columns=[
            'step', 'type', 'amount',
            'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest'
        ])

        proba = model.predict_proba(input_df)[0]

        fraud_probability = proba[1]   
        legit_probability = proba[0]

        FRAUD_THRESHOLD = 0.20

        if fraud_probability >= FRAUD_THRESHOLD:
            result = "Fraud Transaction"
        else:
            result = "Legitimate Transaction"

        return render_template(
            'result.html',
            result=result,
            confidence=round(fraud_probability * 100, 2)
        )

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True) 
    
from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
import math

app = Flask(__name__)

model = xgb.XGBClassifier()
model.load_model("xgb_fraud_detection_model.json")

TYPE_MAPPING = {
    'CASH_IN': 0,
    'CASH_OUT': 1,
    'DEBIT': 2,
    'PAYMENT': 3,
    'TRANSFER': 4
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        step = int(request.form['step'])
        txn_type = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        amount = math.log(amount) if amount > 0 else 0
        txn_type = TYPE_MAPPING[txn_type]

        input_df = pd.DataFrame([[
            step,
            txn_type,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest
        ]], columns=[
            'step', 'type', 'amount',
            'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest'
        ])

        proba = model.predict_proba(input_df)[0]

        fraud_probability = proba[1]   
        legit_probability = proba[0]

        FRAUD_THRESHOLD = 0.20

        if fraud_probability >= FRAUD_THRESHOLD:
            result = "Fraud Transaction"
        else:
            result = "Legitimate Transaction"

        return render_template(
            'result.html',
            result=result,
            confidence=round(fraud_probability * 100, 2)
        )

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
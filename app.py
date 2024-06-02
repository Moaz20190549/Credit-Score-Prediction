from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('your_model.pkl')

# HTML form template with CSS styling
form_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Credit Score Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        form {
            max-width: 600px;
            margin: 0 auto;
            background: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        label {
            font-weight: bold;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            padding: 15px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
            font-size: 16px;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>Credit Score Prediction</h1>
    <h1>0 --> good , 1 --> standerd , 2 --> poor </h1>
    <form action="/predict" method="post">
        <label for="Annual_Income">Annual Income:</label>
        <input type="text" id="Annual_Income" name="Annual_Income"><br><br>
        <label for="Num_Bank_Accounts">Num Bank Accounts:</label>
        <input type="text" id="Num_Bank_Accounts" name="Num_Bank_Accounts"><br><br>
        <label for="Num_Credit_Card">Num Credit Card:</label>
        <input type="text" id="Num_Credit_Card" name="Num_Credit_Card"><br><br>
        <label for="Interest_Rate">Interest Rate:</label>
        <input type="text" id="Interest_Rate" name="Interest_Rate"><br><br>
        <label for="Num_of_Loan">Num of Loan:</label>
        <input type="text" id="Num_of_Loan" name="Num_of_Loan"><br><br>
        <label for="Delay_from_due_date">Delay from Due Date:</label>
        <input type="text" id="Delay_from_due_date" name="Delay_from_due_date"><br><br>
        <label for="Num_of_Delayed_Payment">Num of Delayed Payment:</label>
        <input type="text" id="Num_of_Delayed_Payment" name="Num_of_Delayed_Payment"><br><br>
        <label for="Changed_Credit_Limit">Changed Credit Limit:</label>
        <input type="text" id="Changed_Credit_Limit" name="Changed_Credit_Limit"><br><br>
        <label for="Num_Credit_Inquiries">Num Credit Inquiries:</label>
        <input type="text" id="Num_Credit_Inquiries" name="Num_Credit_Inquiries"><br><br>
        <label for="Outstanding_Debt">Outstanding Debt:</label>
        <input type="text" id="Outstanding_Debt" name="Outstanding_Debt"><br><br>
        <label for="Credit_History_Age">Credit History Age:</label>
        <input type="text" id="Credit_History_Age" name="Credit_History_Age"><br><br>
        <label for="Total_EMI_per_month">Total EMI per Month:</label>
        <input type="text" id="Total_EMI_per_month" name="Total_EMI_per_month"><br><br>
        <label for="Total_Num_Accounts">Total Num Accounts:</label>
        <input type="text" id="Total_Num_Accounts" name="Total_Num_Accounts"><br><br>
        <label for="Debt_Per_Account">Debt Per Account:</label>
        <input type="text" id="Debt_Per_Account" name="Debt_Per_Account"><br><br>
        <label for="Debt_to_Income_Ratio">Debt to Income Ratio:</label>
        <input type="text" id="Debt_to_Income_Ratio" name="Debt_to_Income_Ratio"><br><br>
        <label for="Delayed_Payments_Per_Account">Delayed Payments Per Account:</label>
        <input type="text" id="Delayed_Payments_Per_Account" name="Delayed_Payments_Per_Account"><br><br>
        <label for="Credit_Mix_Encoded">Credit Mix Encoded:</label>
        <input type="text" id="Credit_Mix_Encoded" name="Credit_Mix_Encoded"><br><br>
        <input type="submit" value="Predict">
    </form>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(form_template)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    for key in data:
        data[key] = [float(data[key])]
    df = pd.DataFrame.from_dict(data)
    
    # Ensure the columns match the trained model's expectations
    required_columns = [
        'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 
        'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 
        'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_History_Age', 'Total_EMI_per_month', 
        'Total_Num_Accounts', 'Debt_Per_Account', 'Debt_to_Income_Ratio', 
        'Delayed_Payments_Per_Account', 'Credit_Mix_Encoded'
    ]

    # Reindex the DataFrame to match the model's expected input
    df = df.reindex(columns=required_columns, fill_value=0)

    # Log the input DataFrame
    print("Input DataFrame:")
    print(df)

    # Make prediction
    prediction = model.predict(df)

    # Log the prediction result
    print("Prediction Result:")
    print(prediction)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

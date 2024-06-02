Data Preprocessing
The preprocessing steps include:

Removing special characters and converting strings to numerical values
Handling missing values
Encoding categorical variables
Feature scaling and normalization
Outlier removal

Model Training:
The model training process includes:


Splitting the data into training and testing sets
Using SMOTE to handle class imbalance
Training a RandomForestClassifier model
Evaluating model performance using metrics like accuracy, confusion matrix, and mean squared error
Model Evaluation
The evaluation involves:

Predicting the credit score on the test set
Calculating accuracy
Plotting the confusion matrix for better visualization


Web Application:
The web application allows users to input financial and demographic data to predict their credit score. It uses a Flask framework and renders an HTML form for input. The model loaded from your_model.pkl is used to make predictions.


HTML Form:
The HTML form collects various inputs such as annual income, number of bank accounts, number of credit cards, etc., and sends them to the Flask backend for prediction.

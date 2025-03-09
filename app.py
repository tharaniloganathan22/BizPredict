from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset from CSV
df = pd.read_csv('datsset.csv')  # Update 'your_dataset.csv' with your dataset filename

# Features and target
X = df[['R&D Spend','Administration','Marketing Spend']]
y = df['Profit']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Define API endpoint for prediction
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    rd_spend = float(request.form['rd_spend'])
    administration = float(request.form['administration'])
    marketing_spend = float(request.form['marketing_spend'])

    # Predict profit
    profit_prediction = model.predict([[rd_spend, administration, marketing_spend]])

    return render_template('result.html', profit=profit_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

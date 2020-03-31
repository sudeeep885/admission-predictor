from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    y_hat = model.predict_proba(final)
    output = f'The probability of your admission is = {y_hat[0][1]: .3f}'
    return render_template('index.html', pred = output)

if __name__ == "__main__":
    app.run(debug=True)
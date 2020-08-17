# Importing essential libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'restaurant-sentiment-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df
    
    input_features = [request.form.values()]
    feature_value = np.array(input_features)
    
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
        #return render_template('index.html', prediction=my_prediction)

    
    df = pd.concat([df,pd.DataFrame({'employee_information':feature_value,'Predicted Review':my_prediction})],ignore_index=True)
    df.to_csv('output_data')
    


if __name__ == '__main__':
	app.run(debug=True)
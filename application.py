# import numpy as np
# import pandas as pd
# import pickle
# from flask import Flask,request,jsonify,render_template
# from sklearn.preprocessing import StandardScaler
# application = Flask(__name__)
# app=application


# ## import ridge and scaler
# ridge_model = pickle.load(open(r'files/ridge.pkl', 'rb'))
# scaler_model = pickle.load(open(r'files/scaler.pkl', 'rb'))

# @app.route("/")
# def hello():
#     return render_template('index.html')

# @app.route("/predict",methods=['Get','POST'])
# def predict():
#     if request.method == 'POST':
#         int_features = [x for x in request.form.values()]
#         final_features = scaler_model.transform(np.array(int_features).reshape(1, -1))
#         prediction = ridge_model.predict(final_features)
#         return render_template('home.html', prediction_text='Predicted Price is {}'.format(prediction))

#         output = round(prediction[0], 2)
#     else:
#         return render_template('home.html')

#     return render_template('index.html', prediction_text='Predicted Price is {}'.format(output))

# if __name__ == "__main__":
#     app.run(host="0.0.0.0")


# import numpy as np
# import pickle
# from flask import Flask, request, render_template
# from sklearn.preprocessing import StandardScaler

# # Flask app initialization
# app = Flask(__name__)

# # Load model and scaler
# ridge_model = pickle.load(open('files/ridge.pkl', 'rb'))
# scaler = pickle.load(open('files/scaler.pkl', 'rb'))

# @app.route("/")
# def index():
#     return render_template('index.html')

# @app.route("/predict", methods=['POST'])
# def predict():
#     try:
#         # Extract input values
#         features = [
#             float(request.form['temperature']),
#             float(request.form['humidity']),
#             float(request.form['wind_speed']),
#             float(request.form.get('rainfall', 0))  # Optional input
#         ]
#         # Scale and predict
#         scaled_features = scaler.transform([features])
#         prediction = ridge_model.predict(scaled_features)
#         return render_template(
#             'index.html',
#             prediction_text=f'Predicted Fire Risk Score: {prediction[0]:.2f}'
#         )
#     except Exception as e:
#         return render_template('index.html', prediction_text=f'Error: {str(e)}')

# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, request, render_template
# import pandas as pd
# import numpy as np
# import pickle
# app = Flask(__name__)


# ridge_model = pickle.load(open('files/ridge2.pkl', 'rb'))
# scaler = pickle.load(open('files/scaler2.pkl', 'rb'))
# # Load data (replace with actual data loading logic)
# data = pd.DataFrame({
#     'Temperature': [22.4, 25.1, 21.8],
#     'RH': [45, 50, 55],
#     'Ws': [10, 12, 8],
#     'Rain': [0, 1, 0],
#     'FFMC': [85.2, 89.1, 82.3],
#     'DMC': [100, 120, 110],
#     'ISI': [5.1, 6.3, 4.8],
#     'Classes': ['Low', 'High', 'Medium'],
#     'Region': ['North', 'South', 'East']
# })

# # Filter relevant columns
# relevant_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
# data = data[relevant_columns]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/results', methods=['POST'])
# def results():
#     # Extract input values
#     user_inputs = {
#         'Temperature': float(request.form['Temperature']),
#         'RH': float(request.form['RH']),
#         'Ws': float(request.form['Ws']),
#         'Rain': float(request.form['Rain']),
#         'FFMC': float(request.form['FFMC']),
#         'DMC': float(request.form['DMC']),
#         'ISI': float(request.form['ISI']),
#         'Region': request.form['Region']
#     }

#     # Example processing (replace with your logic)
#     results = data[data['Region'] == user_inputs['Region']]

#     return render_template('results.html', data=results.to_dict(orient='records'))

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load Ridge model and scaler
ridge_model = pickle.load(open('files/ridge2.pkl', 'rb'))
scaler = pickle.load(open('files/scaler2.pkl', 'rb'))

# Sample data
data = pd.DataFrame({
    'Temperature': [22.4, 25.1, 21.8],
    'RH': [45, 50, 55],
    'Ws': [10, 12, 8],
    'Rain': [0, 1, 0],
    'FFMC': [85.2, 89.1, 82.3],
    'DMC': [100, 120, 110],
    'ISI': [5.1, 6.3, 4.8],
    'Classes': ['Low', 'High', 'Medium'],
    'Region': ['North', 'South', 'East']
})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    # Extract input values
    user_inputs = {
        'Temperature': float(request.form['Temperature']),
        'RH': float(request.form['RH']),
        'Ws': float(request.form['Ws']),
        'Rain': float(request.form['Rain']),
        'FFMC': float(request.form['FFMC']),
        'DMC': float(request.form['DMC']),
        'ISI': float(request.form['ISI']),
        'Region': request.form['Region'],
        'Classes': request.form.get('Classes', 'Low')  # Default to 'Low' if not provided
    }

    # Encode Region and Classes
    region_map = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
    class_map = {'Low': 0, 'Medium': 1, 'High': 2}

    user_inputs['Region'] = region_map.get(user_inputs['Region'], -1)
    user_inputs['Classes'] = class_map.get(user_inputs['Classes'], -1)

    # Convert inputs to DataFrame for preprocessing
    input_df = pd.DataFrame([user_inputs])

    # Ensure column order matches scaler's expectation
    expected_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
    input_df = input_df[expected_columns]

    # Standardize input data
    scaled_inputs = scaler.transform(input_df)

    # Predict using Ridge model
    prediction = ridge_model.predict(scaled_inputs)

    # Retrieve matching rows from data for display
    decoded_region = {v: k for k, v in region_map.items()}.get(user_inputs['Region'], 'Unknown')
    results = data[data['Region'] == decoded_region]

    return render_template(
        'results.html',
        data=results.to_dict(orient='records'),
        prediction=prediction[0]  # Display the first prediction result
    )

if __name__ == '__main__':
    app.run(debug=True)

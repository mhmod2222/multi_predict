import numpy as np
from flask import Flask, request, render_template
import pickle


def format_result(result):
    formatted_result = []
    for i in range(result.shape[0]):
        formatted_result.append(str(round(result[i])))
    return formatted_result


def feature_engineering(features):
    final_features = np.empty(19)
    final_features[:3]    = np.array([features[0], (features[0]+features[1])/2.0, features[1]])  # (Temp_H, Temp_L):             --> Temp_H, Temp_avg, Temp_L
    final_features[3:6]   = np.array([features[2], (features[2]+features[3])/2.0, features[3]])  # (Dew_H, Dew_L):               --> Dew_H, Dew_avg, Dew_L
    final_features[6:9]   = np.array([features[4], (features[4]+features[5])/2.0, features[5]])  # (Humidity_H, Humidity_L):     --> Humidity_H, Humidity_avg, Humidity_L
    final_features[9:12]  = np.array([features[6], (features[6]+features[7])/2.0, features[7]])  # (Pressure_H, Pressure_L):     --> Pressure_H, Pressure_avg, Pressure_L
    final_features[12:15] = np.array([features[8], (features[8]+features[9])/2.0, features[9]])  # (Visibility_H, Visibility_L): --> Visibility_H, Visibility_avg, Visibility_L
    final_features[15:17] = np.array([features[10], (features[10]+features[11])/2.0])            # (Wind_H, Wind_L):             --> Wind_H, Wind_avg

    if features[12] == 1:
        final_features[17:] = np.array([1, 0])                                                   # ("sunny"):  --> neutral, rain
    elif features[12] == 2:
        final_features[17:] = np.array([0, 1])                                                   # ("rainy"):  --> neutral, rain
    elif features[12] == 3:
        final_features[17:] = np.array([0, 0])                                                   # ("foggy"):  --> neutral, rain

    return final_features


my_app = Flask(__name__)

model   = pickle.load(open('model.pkl', 'rb'))
scalerX = pickle.load(open('scalerX.pkl', 'rb'))
scalerY = pickle.load(open('scalerY.pkl', 'rb'))


@my_app.route('/')
def home():
    return render_template('home.html')


@my_app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = feature_engineering(features)
    scaled_test_point = scalerX.transform(final_features.reshape(1, -1))
    result = model.predict(scaled_test_point)
    result1 = scalerY.inverse_transform(result)


    formatted_result = format_result(result1[0])

    I = ['Ferrisia Virgata', 'Icerya Seychellarum', 'Icerya purchasi', 'Planococcus Citri',
         'Scymnus Syriacus', 'Cydonia Vicina', 'Chrysoperla Carnea', 'Rodalia',
         'Homalotylus Vicinus', 'Homalotyloidea', 'Leptomastix', 'Leptomastidae',
         'Gyranusoidea Indica', 'Aenasius', 'Chartocerus Subaeneus']

    final_text = ''
    for x in range(len(I)):
        final_text += "{}: {}\n".format(I[x], formatted_result[x])

    return render_template('home.html', prediction_text=final_text)


if __name__ == '__main__':
    my_app.run(debug=True)
    #my_app.run(host='0.0.0.0', port=8080)

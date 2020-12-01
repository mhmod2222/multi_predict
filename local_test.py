import numpy as np
import pickle

# Temp_H, Temp_avg, Temp_L,
# Dew_H, Dew_avg, Dew_L,
# Humidity_H, Humidity_avg, Humidity_L,
# Pressure_H, Pressure_avg, Pressure_L,
# Visibility_H, Visibility_avg, Visibility_L,
# Wind_H, Wind_avg
# neutral, rain

features = [20.0, 11.0, 12.0, 8.0, 88.0, 46.0, 1026.0, 1022.0, 10.0, 2.0, 19.0, 8.0, 1]

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

print(final_features)

model   = pickle.load(open('model.pkl', 'rb'))
scalerX = pickle.load(open('scalerX.pkl', 'rb'))
scalerY = pickle.load(open('scalerY.pkl', 'rb'))

scaled_test_point = scalerX.transform(final_features.reshape(1, -1))
result = model.predict(scaled_test_point)
result1 = scalerY.inverse_transform(result)
print(result1)

#prediction = model.predict(final_features)


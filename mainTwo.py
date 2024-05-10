# import pickle # it helps to load the model
# import numpy as np
# from sklearn.preprocessing import StandardScaler


# # model = pickle.load(open("./HeartAttacks/RFModelPredict.pkl","rb")) # open model in read mode
# model = pickle.load(open("./HeartAttacks/XGBoostModelPredict.pkl","rb")) # open model in read mode

# # Start Preprocessing
# standard_to = StandardScaler(); 
# def predict():
#     age = float(58);
#     gender = float(1);
#     impulse = float(74);
#     pressureHigh = float(124)
#     pressureLow = float(72);
#     glucose = float(116)
#     kcm = float(15);
#     troponin = float(0.4);


#     prediction = model.predict(np.array([[age,gender,impulse,pressureHigh,pressureLow,glucose,kcm,troponin]]).reshape(1,8))

#     output = round(prediction[0],2) # Predict the model with condition

#     if output == 0: # Condition for output
#         print("No Heart Attack is to be expected")# Connect ot html page and app
#     else:
#         pred = "The Patient has Heart Disease ".format(output)
#         print("You might have a Heert attack in the future");


# predict();



import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
model = pickle.load(open("./HeartAttacks/randomForest.pkl","rb")) # open model in read mode

# Start Preprocessing
standard_to = StandardScaler()

def predict():
    # Define input features with named columns
    # input_data = np.array([[58, 1, 74, 124, 72, 116, 15, 0.4]])
    input_data = np.array([[5, 0, 90, 100, 60, 80, 30, 0.1]])

    feature_names = ['age', 'gender', 'impulse', 'pressureHigh', 'pressureLow', 'glucose', 'kcm', 'troponin']

    # Scale the input data
    input_scaled = standard_to.fit_transform(input_data)

    # Predict using the model
    prediction = model.predict(input_scaled)

    # Output the prediction with feature names
    if prediction == 0:
        print("No Heart Attack is to be expected")
    else:
        print("You might have a Heart attack in the future")

    # Print feature names and values
    print("Feature Names:", feature_names)
    print("Input Values:", input_data.flatten())

predict()


# import pickle

# try:
#     model = pickle.load(open("./HeartAttacks/randomForest.pkl", "rb"))
#     # Rest of your code for preprocessing and prediction
# except EOFError:
#     print("Error: Failed to load model. The file may be empty or corrupted.")


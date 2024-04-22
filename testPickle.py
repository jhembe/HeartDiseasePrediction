import pickle

with open("Logistic_regression_model.pkl", "rb") as f:
    try:
        data = pickle.load(f)
        print(data)
    except EOFError:
        print("The file is empty or doesn't contain valid pickled data.")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from src.preprocessing import preprocessing_data
import joblib

x_train,x_test,y_train,y_test = preprocessing_data("data/data.csv")

def train_DecisionTree():
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = round(accuracy_score(y_test, pred)*100,2)
    joblib.dump(model, "models/decision_tree_model.pkl")
    return model, acc
    

def train_GradientBoosting():
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = round(accuracy_score(y_test, pred)*100,2)
    return model, acc
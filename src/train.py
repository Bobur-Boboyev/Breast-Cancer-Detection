from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing import preprocessing_data

(x_train,x_test,y_train,y_test), scaler = preprocessing_data("data/data.csv")

def train_DecisionTree():
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = round(accuracy_score(y_test, pred)*100,2)
    cm = confusion_matrix(y_test, pred)
    print(f"\nConfusion Matrix of Decision Tree:\n\n{cm}\n")
    return f"accuracy of Decision Tree: %{acc}\n"

def train_GradientBoosting():
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = round(accuracy_score(y_test, pred)*100,2)
    cm = confusion_matrix(y_test, pred)
    print(f"\nConfusion Matrix of Gradient Boosting:\n\n{cm}\n")
    return f"accuracy of Gradient Boosting: %{acc}"


print(train_DecisionTree())
print(train_GradientBoosting())
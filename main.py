from src.train import train_DecisionTree, train_GradientBoosting
from src.predict import predict

while True:
    print("\n=== Breast Cancer Detection Menu ===")
    print("1 - Decision Tree Accuracy")
    print("2 - Gradient Boosting Accuracy")
    print("3 - Predict Diagnosis")
    print("0 - Exit")

    choice = input("\nEnter your choice: ")

    if choice == '1':
        model, acc = train_DecisionTree()
        print(f"\nAccuracy of Decision Tree: {acc}%")
    
    elif choice == '2':
        model, acc = train_GradientBoosting()
        print(f"\nAccuracy of Gradient Boosting: {acc}%")
    
    elif choice == '3':
        result = predict()
        if result:
            print(result)
    
    elif choice == '0':
        print("Exiting program.")
        break

    else:
        print("Wrong input. Please try again.")

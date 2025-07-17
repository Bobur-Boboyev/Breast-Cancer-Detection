import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def predict():
    print("Please enter the following 30 values for the patient:\n")
    
    try:
        radius_mean = float(input("1. radius_mean: "))
        texture_mean = float(input("2. texture_mean: "))
        perimeter_mean = float(input("3. perimeter_mean: "))
        area_mean = float(input("4. area_mean: "))
        smoothness_mean = float(input("5. smoothness_mean: "))
        compactness_mean = float(input("6. compactness_mean: "))
        concavity_mean = float(input("7. concavity_mean: "))
        concave_points_mean = float(input("8. concave points_mean: "))
        symmetry_mean = float(input("9. symmetry_mean: "))
        fractal_dimension_mean = float(input("10. fractal_dimension_mean: "))
        radius_se = float(input("11. radius_se: "))
        texture_se = float(input("12. texture_se: "))
        perimeter_se = float(input("13. perimeter_se: "))
        area_se = float(input("14. area_se: "))
        smoothness_se = float(input("15. smoothness_se: "))
        compactness_se = float(input("16. compactness_se: "))
        concavity_se = float(input("17. concavity_se: "))
        concave_points_se = float(input("18. concave points_se: "))
        symmetry_se = float(input("19. symmetry_se: "))
        fractal_dimension_se = float(input("20. fractal_dimension_se: "))
        radius_worst = float(input("21. radius_worst: "))
        texture_worst = float(input("22. texture_worst: "))
        perimeter_worst = float(input("23. perimeter_worst: "))
        area_worst = float(input("24. area_worst: "))
        smoothness_worst = float(input("25. smoothness_worst: "))
        compactness_worst = float(input("26. compactness_worst: "))
        concavity_worst = float(input("27. concavity_worst: "))
        concave_points_worst = float(input("28. concave points_worst: "))
        symmetry_worst = float(input("29. symmetry_worst: "))
        fractal_dimension_worst = float(input("30. fractal_dimension_worst: "))
        
    except ValueError:
        print("Please enter only a number.")
        return

    input_features = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]

    model = joblib.load("models/decision_tree_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    input_array = np.array(input_features).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    pred = model.predict(scaled_input)[0]

    return f'\nTumor is predicted to be **{"M" if pred == 1 else "B"}**'

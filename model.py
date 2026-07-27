
import joblib
import pandas as pd

# Load model and preprocessor only once
model = joblib.load("decision_tree.pkl")
preprocessor = joblib.load("preprocessor.pkl")


def predict_math_score(
    gender,
    race,
    parental_education,
    lunch,
    test_preparation,
    reading_score,
    writing_score
):
    """
    Predict Math Score
    """

    data = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [parental_education],
        "lunch": [lunch],
        "test preparation course": [test_preparation],
        "reading score": [reading_score],
        "writing score": [writing_score]
    })

    processed_data = preprocessor.transform(data)

    prediction = model.predict(processed_data)

    return round(float(prediction[0]), 2)


# Test
if __name__ == "__main__":

    result = predict_math_score(
        gender="female",
        race="group B",
        parental_education="bachelor's degree",
        lunch="standard",
        test_preparation="completed",
        reading_score=80,
        writing_score=82
    )

    print("Predicted Math Score:", result)

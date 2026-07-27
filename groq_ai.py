import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """
You are an AI Assistant for a Student Performance Prediction project.

You answer only questions related to:
- Machine Learning
- Decision Tree Regressor
- Student Performance Prediction
- EDA
- Dataset
- R2 Score
- MAE
- RMSE
- Python
- Streamlit

Keep answers simple and concise.
"""

def ask_ai(question):

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],

        temperature=0.3
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    while True:

        q = input("You : ")

        if q.lower() == "exit":
            break

        print()

        print("AI :", ask_ai(q))

        print()

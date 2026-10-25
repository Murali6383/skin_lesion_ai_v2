import requests


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "llama3.2"


def generate_response(
    disease: str,
    confidence: float,
    user_prompt: str
):

    system_prompt = """
You are a medical information assistant inside a skin lesion
AI screening application.

A separate skin-lesion image classification model has analyzed
the uploaded image.

Your job is to explain the AI result according to the user's
question.

IMPORTANT SAFETY RULES:

1. The AI prediction is NOT a confirmed medical diagnosis.
2. Give general medical information only.
3. Do not prescribe medicines.
4. Do not provide medication dosage.
5. Do not tell the user to start or stop a medicine.
6. Do not claim certainty.
7. Explain relevant safety precautions.
8. Mention warning signs when relevant.
9. If treatment is asked, explain general treatment approaches.
10. Recommend consultation with a qualified dermatologist when
    appropriate.
11. Answer the user's actual question directly.
12. Use simple language that a normal user can understand.

The image model prediction should be treated as an AI screening
result, not as a final diagnosis.
"""

    user_message = f"""
IMAGE MODEL RESULT
------------------
Predicted class/disease: {disease}
Model confidence: {confidence:.2f}%

USER QUESTION
-------------
{user_prompt}

TASK
----
Answer the user's question using the image-model result as
context.

If relevant, include:
- What the predicted condition means
- Safety precautions
- Warning signs
- General treatment information
- When professional medical evaluation is appropriate

Do not present the prediction as a confirmed diagnosis.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]
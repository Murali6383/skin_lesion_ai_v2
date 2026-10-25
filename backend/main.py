import os
import json
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

# =========================================================
# FASTAPI
# =========================================================

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

# =========================================================
# GROQ
# =========================================================

from groq import Groq

# =========================================================
# PROJECT CONFIG
# =========================================================

from ai.config import (
    CLASS_NAMES,
    RESULTS_DIR,
)

# =========================================================
# EXISTING DISEASE MODEL
# =========================================================

from ai.predict import predictor

# =========================================================
# SKIN IMAGE VALIDATOR
# =========================================================

from backend.image_validator import validate_skin_image

# =========================================================
# NORMAL SKIN DETECTOR
# =========================================================

from ai.normal_skin_detector.detector import detect_normal_skin

# =========================================================
# GRAD-CAM
# =========================================================

from ai.gradcam.generate_gradcam import generate_gradcam


# =========================================================
# GROQ CONFIGURATION
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "openai/gpt-oss-20b"

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set.")

groq_client = None

if GROQ_API_KEY:
    groq_client = Groq(
        api_key=GROQ_API_KEY
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="SkinGuardian AI + Groq",
    version="5.1.0",
)


# =========================================================
# RESULTS / GRAD-CAM STATIC FILES
# =========================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

app.mount(
    "/results",
    StaticFiles(
        directory=str(RESULTS_DIR)
    ),
    name="results",
)


# =========================================================
# CORS
# =========================================================

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

FRONTEND_DIR = (
    BASE_DIR / "frontend"
)

FRONTEND_DIST = (
    FRONTEND_DIR / "dist"
)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "SkinGuardian AI + Groq",
        "version": "5.1.0",
        "groq_model": GROQ_MODEL,
        "groq_configured": bool(GROQ_API_KEY),
        "skin_validator": True,
        "normal_skin_detector": True,
        "disease_model": True,
        "gradcam": True,
    }


# =========================================================
# GROQ TEST
# =========================================================

@app.get("/groq-test")
def groq_test():

    if not GROQ_API_KEY or groq_client is None:

        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured.",
        )

    try:

        response = (
            groq_client
            .chat
            .completions
            .create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Say hello and confirm "
                            "that you are running "
                            "through Groq."
                        ),
                    }
                ],
                max_completion_tokens=50,
                include_reasoning=False,
            )
        )

        return {
            "success": True,
            "model": GROQ_MODEL,
            "response": (
                response
                .choices[0]
                .message
                .content
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Groq connection failed: {str(e)}",
        )


# =========================================================
# IMAGE EXTENSION VALIDATION
# =========================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def validate_image(
    filename: Optional[str],
):

    ext = Path(
        filename or ""
    ).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Please upload JPG, JPEG, PNG, "
                "BMP or WEBP."
            ),
        )

    return ext


# =========================================================
# SKIN IMAGE GATE
# =========================================================

def validate_skin_content(
    image_path: str,
):

    print()
    print("------------------------------------------")
    print("STEP 1: RUNNING SKIN IMAGE VALIDATOR")
    print("------------------------------------------")

    result = validate_skin_image(
        image_path
    )

    print(
        "Validator result:",
        result,
    )

    # -----------------------------------------------------
    # INVALID IMAGE
    # -----------------------------------------------------

    if not result.get(
        "valid_image",
        False,
    ):

        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_IMAGE",
                "message": result.get(
                    "message",
                    "Unable to read image.",
                ),
            },
        )

    # -----------------------------------------------------
    # NON-SKIN IMAGE
    # -----------------------------------------------------

    if not result.get(
        "is_skin",
        False,
    ):

        print()
        print("❌ NON-SKIN IMAGE DETECTED")
        print("❌ AI DISEASE MODEL WILL NOT RUN")

        raise HTTPException(
            status_code=400,
            detail={
                "error": "NON_SKIN_IMAGE",
                "message": (
                    "This is not a skin image. "
                    "Please upload a valid skin "
                    "or dermoscopic image."
                ),
                "detected_class": result.get(
                    "class",
                    "non_skin",
                ),
                "confidence_percent": result.get(
                    "confidence",
                    0,
                ),
                "distance": result.get(
                    "distance",
                    0,
                ),
                "threshold": result.get(
                    "threshold",
                    0,
                ),
                "decision": result.get(
                    "decision",
                    "REJECT",
                ),
            },
        )

    # -----------------------------------------------------
    # SKIN IMAGE ACCEPTED
    # -----------------------------------------------------

    print()
    print("✅ SKIN IMAGE ACCEPTED")

    return result


# =========================================================
# NORMAL SKIN RESULT HELPER
# =========================================================

def build_normal_skin_response(
    validation,
    normal_result,
):

    return {

        "success": True,

        "validation": {

            "is_skin": validation.get(
                "is_skin",
                True,
            ),

            "class": validation.get(
                "class",
                "possible_skin",
            ),

            "decision": validation.get(
                "decision",
                "PASS",
            ),

            "distance": validation.get(
                "distance",
                0,
            ),

            "threshold": validation.get(
                "threshold",
                0,
            ),

            "validator_type": validation.get(
                "validator_type",
                "non_skin_nearest_neighbour",
            ),
        },

        "normal_skin": normal_result,

        "predicted_class": "No Disease Detected",

        "confidence_percent": None,

        "disease_model_ran": False,

        "gradcam_url": None,

        "note": (
            "No clear abnormality was detected "
            "by the research model. This does not "
            "confirm that the skin is medically healthy."
        ),
    }


# =========================================================
# GRAD-CAM HELPER
# =========================================================

def generate_gradcam_for_prediction(image_path, disease):
    try:
        class_index = CLASS_NAMES.index(disease)

        print("------------------------------------------")
        print("STEP 4: GENERATING GRAD-CAM")
        print("------------------------------------------")

        print(f"Disease: {disease}")
        print(f"Class index: {class_index}")

        gradcam_result = generate_gradcam(
            image_path,
            class_index
        )

        # generate_gradcam() returns a dictionary
        gradcam_path = gradcam_result["path"]

        gradcam_file = Path(gradcam_path)

        gradcam_url = (
            f"/results/gradcam/{gradcam_file.name}"
        )

        print(f"Grad-CAM file: {gradcam_file}")
        print(f"Grad-CAM URL: {gradcam_url}")

        return gradcam_url

    except Exception as error:
        print(
            f"[Grad-CAM] ERROR: {repr(error)}"
        )
        return None

    print(
        "Disease:",
        disease,
    )

    print(
        "Class index:",
        class_index,
    )

    # -----------------------------------------------------
    # GENERATE GRAD-CAM
    # -----------------------------------------------------

    try:

        gradcam_result = generate_gradcam(
            image_path,
            class_index,
        )

        if not gradcam_result:

            print(
                "[Grad-CAM] No output generated."
            )

            return None

        gradcam_file = Path(
            gradcam_result
        )

        print(
            "Grad-CAM output:",
            gradcam_file,
        )

        # -------------------------------------------------
        # URL
        # -------------------------------------------------

        gradcam_url = (
            f"/results/gradcam/"
            f"{gradcam_file.name}"
        )

        print(
            "Grad-CAM URL:",
            gradcam_url,
        )

        return gradcam_url

    except Exception as e:

        print(
            "[Grad-CAM] ERROR:",
            repr(e),
        )

        return None


# =========================================================
# PREDICTION API
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):

    temp_path = None

    try:

        # -------------------------------------------------
        # FILE EXTENSION
        # -------------------------------------------------

        ext = validate_image(
            file.filename
        )

        # -------------------------------------------------
        # READ FILE
        # -------------------------------------------------

        data = await file.read()

        if not data:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        # -------------------------------------------------
        # TEMP IMAGE
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext,
        ) as temp_file:

            temp_file.write(data)

            temp_path = temp_file.name

        print()
        print("==========================================")
        print("PREDICTION REQUEST")
        print("==========================================")

        print(
            "Uploaded file:",
            file.filename,
        )

        print(
            "Temporary image:",
            temp_path,
        )

        # =================================================
        # STEP 1: SKIN VALIDATION
        # =================================================

        validation = validate_skin_content(
            temp_path
        )

        # =================================================
        # STEP 2: NORMAL SKIN DETECTION
        # =================================================

        print()
        print("------------------------------------------")
        print("STEP 2: NORMAL SKIN DETECTOR")
        print("------------------------------------------")

        normal_result = detect_normal_skin(
            temp_path
        )

        print(
            "Normal skin result:",
            normal_result,
        )

        # =================================================
        # NORMAL SKIN
        # =================================================

        if normal_result.get(
            "is_normal",
            False,
        ):

            print()
            print("==========================================")
            print("✅ NORMAL SKIN DETECTED")
            print("==========================================")

            print(
                "Decision:",
                "No Disease Detected",
            )

            return build_normal_skin_response(
                validation,
                normal_result,
            )

        # =================================================
        # STEP 3: EXISTING DISEASE MODEL
        # =================================================

        print()
        print("------------------------------------------")
        print("STEP 3: EXISTING DISEASE MODEL")
        print("------------------------------------------")

        result = predictor.predict(
            temp_path
        )

        disease = str(
            result.get(
                "prediction",
                "Unknown",
            )
        )

        confidence = result.get(
            "confidence",
            0,
        )

        print(
            "Predicted disease:",
            disease,
        )

        print(
            "Confidence:",
            confidence,
        )

        # =================================================
        # STEP 4: GRAD-CAM
        # =================================================

        gradcam_url = (
            generate_gradcam_for_prediction(
                temp_path,
                disease,
            )
        )

        # =================================================
        # DISEASE RESPONSE
        # =================================================

        return {

            "success": True,

            "validation": {

                "is_skin": validation.get(
                    "is_skin",
                    True,
                ),

                "class": validation.get(
                    "class",
                    "possible_skin",
                ),

                "decision": validation.get(
                    "decision",
                    "PASS",
                ),

                "distance": validation.get(
                    "distance",
                    0,
                ),

                "threshold": validation.get(
                    "threshold",
                    0,
                ),

                "validator_type": validation.get(
                    "validator_type",
                    "non_skin_nearest_neighbour",
                ),
            },

            "normal_skin": normal_result,

            "predicted_class": disease,

            "confidence_percent": confidence,

            "disease_model_ran": True,

            "gradcam_url": gradcam_url,

            "note": (
                "Research classification output; "
                "not a medical diagnosis."
            ),
        }

    except HTTPException:

        raise

    except Exception as e:

        print(
            "PREDICTION ERROR:",
            repr(e),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    finally:

        if temp_path:

            try:

                os.remove(
                    temp_path
                )

                print(
                    "Temporary image deleted."
                )

            except OSError:

                pass


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """

You are SkinGuardian AI, a medical information
assistant integrated into a skin lesion AI
screening application.

A separate machine learning system has analyzed
the uploaded skin image.

The result is provided only as contextual
research information.

IMPORTANT RESPONSE RULES:

1. ALWAYS answer the user's CURRENT PROMPT directly.

2. The CURRENT USER PROMPT is the primary instruction.

3. Use the previous AI prediction ONLY when it is
relevant to the current question.

4. Do NOT automatically repeat the prediction,
confidence, symptoms, precautions, or treatment.

5. Do NOT generate the same explanation for every
user message.

6. If the user says:
"hi", "hello", "hii", "hey", or "thanks",
respond naturally and briefly.

7. If the user asks:
"what disease is this?"
explain the AI predicted condition and confidence.
Clearly state that the prediction is NOT a confirmed
medical diagnosis.

8. If the system result says:
"No Disease Detected",
explain that the research model did not detect
a clear abnormality in the reference dataset.
Do NOT claim that the person is definitely healthy.

9. If the user asks:
"why did this disease come?"
explain possible causes and risk factors related
to the predicted condition.

10. If the user asks about symptoms,
explain only the relevant symptoms.

11. If the user asks about treatment,
explain general treatment approaches only.

12. If the user asks for drugs, medicines, tablets,
prescriptions, or dosage:
do NOT prescribe medicines or provide dosage.
Explain that a qualified dermatologist should
confirm the condition and decide appropriate treatment.

13. If the user asks an unrelated question,
answer that question directly.

14. AI predictions are NOT confirmed diagnoses.

15. Provide general medical information only.

16. Do not claim certainty.

17. Do not personally diagnose the patient.

18. Do not analyze the uploaded image yourself.

19. Do not say that you personally examined the image.

20. Do not tell the user to start or stop medication.

21. Recommend consultation with a qualified dermatologist
when medically appropriate.

22. Use simple and clear language.

23. Keep answers concise and directly related
to the user's current question.

24. Use bullet points only when they improve clarity.

25. Never invent patient information.

26. For follow-up questions, use the previous prediction
only when relevant.

"""


# =========================================================
# CHAT ERROR STREAM
# =========================================================

async def chat_error_stream(
    message: str,
):

    yield (
        json.dumps({
            "type": "error",
            "message": message,
        })
        + "\n"
    )


# =========================================================
# CHAT API
# =========================================================

@app.post("/chat")
async def chat(

    file: Optional[UploadFile] = File(None),

    prompt: str = Form(...),

    first_message: bool = Form(True),

    previous_disease: str = Form(""),

    previous_confidence: str = Form(""),

):

    print()
    print("==========================================")
    print("SKINGUARDIAN CHAT REQUEST")
    print("==========================================")

    print(
        "Prompt:",
        prompt,
    )

    print(
        "First message:",
        first_message,
        type(first_message),
    )

    print(
        "Previous disease:",
        previous_disease,
    )

    print(
        "Previous confidence:",
        previous_confidence,
    )

    if file:

        print(
            "Uploaded file:",
            file.filename,
        )

    else:

        print(
            "Uploaded file: NONE"
        )

    # =====================================================
    # PROMPT CHECK
    # =====================================================

    if not prompt or not prompt.strip():

        return StreamingResponse(
            chat_error_stream(
                "Please enter a question or prompt."
            ),
            media_type="application/x-ndjson",
        )

    # =====================================================
    # GROQ CHECK
    # =====================================================

    if not GROQ_API_KEY or groq_client is None:

        return StreamingResponse(
            chat_error_stream(
                "GROQ_API_KEY is missing. "
                "Please configure your .env file."
            ),
            media_type="application/x-ndjson",
        )

    # =====================================================
    # VARIABLES
    # =====================================================

    disease = ""

    confidence = None

    validation_result = None

    normal_skin_result = None

    gradcam_url = None

    temp_path = None

    try:

        # =================================================
        # FIRST MESSAGE
        # =================================================

        if first_message:

            print()
            print("MODE: FIRST MESSAGE")

            # ---------------------------------------------
            # IMAGE REQUIRED
            # ---------------------------------------------

            if file is None:

                return StreamingResponse(
                    chat_error_stream(
                        "Please upload a skin image "
                        "before starting the analysis."
                    ),
                    media_type="application/x-ndjson",
                )

            # ---------------------------------------------
            # FILE EXTENSION
            # ---------------------------------------------

            try:

                ext = validate_image(
                    file.filename
                )

            except HTTPException as e:

                return StreamingResponse(
                    chat_error_stream(
                        str(e.detail)
                    ),
                    media_type="application/x-ndjson",
                )

            # ---------------------------------------------
            # READ FILE
            # ---------------------------------------------

            data = await file.read()

            if not data:

                return StreamingResponse(
                    chat_error_stream(
                        "Uploaded image is empty."
                    ),
                    media_type="application/x-ndjson",
                )

            # ---------------------------------------------
            # TEMP IMAGE
            # ---------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=ext,
            ) as temp_file:

                temp_file.write(data)

                temp_path = temp_file.name

            print(
                "Temporary image:",
                temp_path,
            )

            # =============================================
            # STEP 1: SKIN VALIDATION
            # =============================================

            print()
            print("------------------------------------------")
            print("STEP 1: SKIN VALIDATION")
            print("------------------------------------------")

            try:

                validation_result = (
                    validate_skin_content(
                        temp_path
                    )
                )

            except HTTPException as e:

                detail = e.detail

                print()
                print("❌ IMAGE REJECTED")

                if isinstance(
                    detail,
                    dict,
                ):

                    message = detail.get(
                        "message",
                        "Invalid image.",
                    )

                else:

                    message = str(detail)

                return StreamingResponse(
                    chat_error_stream(
                        message
                    ),
                    media_type="application/x-ndjson",
                )

            # =============================================
            # STEP 2: NORMAL SKIN DETECTOR
            # =============================================

            print()
            print("------------------------------------------")
            print("STEP 2: NORMAL SKIN DETECTOR")
            print("------------------------------------------")

            try:

                normal_skin_result = detect_normal_skin(
                    temp_path
                )

            except Exception as e:

                print(
                    "NORMAL SKIN DETECTOR ERROR:",
                    repr(e),
                )

                return StreamingResponse(
                    chat_error_stream(
                        "Skin validation passed, "
                        "but the normal-skin detector "
                        "failed. Please check whether "
                        "normal_skin_detector.pth exists."
                    ),
                    media_type="application/x-ndjson",
                )

            print(
                "Normal skin result:",
                normal_skin_result,
            )

            # =============================================
            # NORMAL SKIN
            # =============================================

            if normal_skin_result.get(
                "is_normal",
                False,
            ):

                print()
                print("==========================================")
                print("✅ NORMAL SKIN DETECTED")
                print("==========================================")

                disease = "No Disease Detected"

                confidence = None

                gradcam_url = None

            # =============================================
            # NOT NORMAL → DISEASE MODEL
            # =============================================

            else:

                print()
                print("------------------------------------------")
                print("STEP 3: EXISTING DISEASE MODEL")
                print("------------------------------------------")

                try:

                    print(
                        "Running existing disease model..."
                    )

                    result = predictor.predict(
                        temp_path
                    )

                except Exception as e:

                    print(
                        "PREDICTOR ERROR:",
                        repr(e),
                    )

                    return StreamingResponse(
                        chat_error_stream(
                            "Image passed skin validation, "
                            "but the AI prediction failed."
                        ),
                        media_type="application/x-ndjson",
                    )

                disease = str(
                    result.get(
                        "prediction",
                        "Unknown",
                    )
                )

                try:

                    confidence = float(
                        result.get(
                            "confidence",
                            0,
                        )
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    confidence = 0.0

                print()
                print(
                    "✅ DISEASE PREDICTION COMPLETED"
                )

                print(
                    "Predicted disease:",
                    disease,
                )

                print(
                    "Confidence:",
                    confidence,
                )

                # =========================================
                # STEP 4: GRAD-CAM
                # =========================================

                gradcam_url = (
                    generate_gradcam_for_prediction(
                        temp_path,
                        disease,
                    )
                )

        # =================================================
        # FOLLOW-UP MESSAGE
        # =================================================

        else:

            print()
            print("MODE: FOLLOW-UP")

            if not previous_disease.strip():

                return StreamingResponse(
                    chat_error_stream(
                        "Previous prediction information "
                        "was not provided."
                    ),
                    media_type="application/x-ndjson",
                )

            disease = previous_disease.strip()

            # ---------------------------------------------
            # NORMAL RESULT
            # ---------------------------------------------

            if disease == "No Disease Detected":

                confidence = None

            # ---------------------------------------------
            # DISEASE RESULT
            # ---------------------------------------------

            else:

                try:

                    confidence = float(
                        previous_confidence
                        if previous_confidence
                        else 0
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    confidence = 0.0

            print(
                "Previous disease:",
                disease,
            )

            print(
                "Previous confidence:",
                confidence,
            )

            # Follow-up does not regenerate Grad-CAM.
            gradcam_url = None

        # =================================================
        # CONFIDENCE TEXT
        # =================================================

        if confidence is None:

            confidence_text = "Not applicable"

        else:

            confidence_text = (
                f"{confidence:.2f}%"
            )

        # =================================================
        # GROQ USER MESSAGE
        # =================================================

        user_message = f"""

Previous AI Skin Analysis Result:
{disease}

AI Confidence:
{confidence_text}

CURRENT USER PROMPT:
{prompt}

IMPORTANT:

Answer the CURRENT USER PROMPT directly.

Use the previous AI result only when relevant.

Do not automatically repeat the result,
confidence, symptoms, precautions, or treatment.

If the previous result is "No Disease Detected",
do not claim that the person is definitely healthy.

The AI result is not a confirmed medical diagnosis.

"""

        # =================================================
        # STREAM GENERATOR
        # =================================================

        async def generate():

            try:

                # =========================================
                # PREDICTION EVENT
                # =========================================

                prediction_event = {

                    "type": "prediction",

                    "disease": disease,

                    "confidence": confidence,

                    "prediction_performed": first_message,

                    "gradcam_url": (
                        gradcam_url
                        if first_message
                        else None
                    ),
                }

                # =========================================
                # VALIDATION INFORMATION
                # =========================================

                if (
                    first_message
                    and validation_result is not None
                ):

                    prediction_event["validation"] = {

                        "is_skin": validation_result.get(
                            "is_skin",
                            True,
                        ),

                        "class": validation_result.get(
                            "class",
                            "possible_skin",
                        ),

                        "decision": validation_result.get(
                            "decision",
                            "PASS",
                        ),

                        "distance": validation_result.get(
                            "distance",
                            0,
                        ),

                        "threshold": validation_result.get(
                            "threshold",
                            0,
                        ),

                        "validator_type": validation_result.get(
                            "validator_type",
                            "non_skin_nearest_neighbour",
                        ),
                    }

                # =========================================
                # NORMAL SKIN INFORMATION
                # =========================================

                if (
                    first_message
                    and normal_skin_result is not None
                ):

                    prediction_event["normal_skin"] = {

                        "is_normal": normal_skin_result.get(
                            "is_normal",
                            False,
                        ),

                        "class": normal_skin_result.get(
                            "class",
                            "unknown",
                        ),

                        "decision": normal_skin_result.get(
                            "decision",
                            "UNKNOWN",
                        ),

                        "distance": normal_skin_result.get(
                            "distance",
                            0,
                        ),

                        "threshold": normal_skin_result.get(
                            "threshold",
                            0,
                        ),

                        "normality_percent": normal_skin_result.get(
                            "normality_percent",
                            0,
                        ),

                        "detector_type": normal_skin_result.get(
                            "detector_type",
                            "normal_skin_nearest_neighbour",
                        ),
                    }

                # =========================================
                # SEND PREDICTION
                # =========================================

                yield (
                    json.dumps(
                        prediction_event
                    )
                    + "\n"
                )

                # =========================================
                # GROQ STREAM
                # =========================================

                print(
                    "Sending request to Groq..."
                )

                stream = (
                    groq_client
                    .chat
                    .completions
                    .create(

                        model=GROQ_MODEL,

                        messages=[

                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT,
                            },

                            {
                                "role": "user",
                                "content": user_message,
                            },

                        ],

                        temperature=0.2,

                        max_completion_tokens=400,

                        include_reasoning=False,

                        stream=True,
                    )
                )

                # =========================================
                # STREAM TOKENS
                # =========================================

                for chunk in stream:

                    if not chunk.choices:

                        continue

                    delta = (
                        chunk
                        .choices[0]
                        .delta
                    )

                    if not delta:

                        continue

                    content = delta.content

                    if content:

                        yield (
                            json.dumps({

                                "type": "token",

                                "content": content,

                            })
                            + "\n"
                        )

                # =========================================
                # DONE
                # =========================================

                yield (
                    json.dumps({
                        "type": "done",
                    })
                    + "\n"
                )

                print(
                    "Groq response completed."
                )

            except Exception as e:

                print(
                    "GROQ STREAM ERROR:",
                    repr(e),
                )

                yield (
                    json.dumps({

                        "type": "error",

                        "message":
                            f"Groq error: {str(e)}",

                    })
                    + "\n"
                )

        # =================================================
        # RETURN STREAM
        # =================================================

        return StreamingResponse(

            generate(),

            media_type="application/x-ndjson",

            headers={

                "Cache-Control": "no-cache",

                "X-Accel-Buffering": "no",

                "Connection": "keep-alive",
            },
        )

    # =====================================================
    # UNEXPECTED ERROR
    # =====================================================

    except Exception as e:

        print(
            "CHAT ERROR:",
            repr(e),
        )

        return StreamingResponse(

            chat_error_stream(
                f"AI processing failed: {str(e)}"
            ),

            media_type="application/x-ndjson",
        )

    finally:

        # -------------------------------------------------
        # DELETE TEMP IMAGE
        # -------------------------------------------------

        if temp_path:

            try:

                os.remove(
                    temp_path
                )

                print(
                    "Temporary image deleted."
                )

            except OSError:

                pass


# =========================================================
# FRONTEND STATIC FILES
# =========================================================

if FRONTEND_DIST.exists():

    assets_dir = (
        FRONTEND_DIST / "assets"
    )

    if assets_dir.exists():

        app.mount(
            "/assets",
            StaticFiles(
                directory=assets_dir
            ),
            name="assets",
        )


# =========================================================
# FRONTEND HOME
# =========================================================

@app.get("/")
async def home():

    index_file = (
        FRONTEND_DIST
        / "index.html"
    )

    if index_file.exists():

        return FileResponse(
            index_file
        )

    return {

        "message":
            "SkinGuardian AI Backend is running.",

        "frontend":
            "Vite frontend is not built yet.",

        "development":
            "Run npm run dev inside the frontend folder.",

        "production":
            "Run npm run build inside the frontend folder.",

        "health":
            "/health",

        "docs":
            "/docs",
    }


# =========================================================
# SPA FALLBACK
# =========================================================

@app.get("/{full_path:path}")
async def frontend_fallback(
    full_path: str,
):

    api_paths = [
        "health",
        "groq-test",
        "predict",
        "chat",
        "docs",
        "openapi.json",
    ]

    first_part = (
        full_path.split("/")[0]
    )

    if first_part in api_paths:

        raise HTTPException(
            status_code=404,
            detail="Not found",
        )

    index_file = (
        FRONTEND_DIST
        / "index.html"
    )

    if index_file.exists():

        return FileResponse(
            index_file
        )

    raise HTTPException(
        status_code=404,
        detail="Frontend not found",
    )


# =========================================================
# LOCAL SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )

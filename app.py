from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import numpy as np

app = FastAPI()

# Templates folder
templates = Jinja2Templates(directory="templates")

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Home Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )


# Prediction Route
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Pregnancies: int = Form(...),
    Glucose: float = Form(...),
    BloodPressure: float = Form(...),
    SkinThickness: float = Form(...),
    Insulin: float = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...)
):

    features = np.array([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]])

    prediction = model.predict(features)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": result
        }
    )
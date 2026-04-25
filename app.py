from fastapi import FastAPI

app = FastAPI(title="notes2latex-ocr")


@app.post("/predict")
async def predict():
    raise NotImplementedError


@app.post("/predict-page")
async def predict_page():
    raise NotImplementedError

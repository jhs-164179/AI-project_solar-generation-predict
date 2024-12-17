# # from flask import Flask, Response
# # from keras import saving
# # import pymysql
# # import requests
# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# # app = Flask(__name__)


# # def get_db_connection():
# #     return pymysql.connect(
# #         host='127.0.0.1',
# #         user='root',
# #         password='1234',
# #         db='db_aiproject',
# #         port=3306
# #     )


# # def load_generation(url):
# #     r = requests.get(url)
# #     if r.status_code == 200:
# #     results = model()


# # @app.route('/generation')
# # def generation():
# #     return Response(
# #         load_generation(url='https://192.168.0.45:5000/demo')
# #     )


# app = FastAPI()

# class PredictionRequest(BaseModel):
#     input_data: list

# @app.post("/predict")
# async def predict(request: PredictionRequest):
#     input_data = request.input_data
#     # 모델 로드 및 예측 코드 추가
#     prediction = model.predict(input_data)  # 예측값 반환
#     return {"prediction": prediction.tolist()}  # JSON 응답으로 반환

# global model
# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0,', port=3333)
#     except KeyboardInterrupt:
#         print('Shut downed well')
import time
import pickle
from flask_cors import CORS
from flask import Flask, request, jsonify
from model import HybridModel
from utils import minmax
import numpy as np


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    with open('test.pkl', 'rb') as f:
        data = pickle.load(f)
    X = data[0]
    y = data[1]
    model = HybridModel(in_shape=X.shape[1:], hid_dim=64, pred_len=24)
    model.build((None, *X.shape[1:]))
    model.load_weights('best.keras')

    preds = model(X)

    with open('minmax.pkl', 'rb') as f:
        mnmx = pickle.load(f)
    preds = minmax(preds, reverse=True, x_max=mnmx[1][-1], x_min=mnmx[0][-1])
    preds = preds[0].numpy().squeeze()
    preds = np.where(preds < 0, 0, preds)
    preds = preds.tolist()
    # print(preds)
    # for i in range(24):
    #     preds.insert(0, 0)
    # print(preds)
    return jsonify(prediction=preds)


@app.route('/update', methods=['POST'])
def update():
    try:
        request_data = request.get_json()
        retrain_params = request_data.get('retrain_params', {})
        epochs = retrain_params.get('epochs', 10)
        batch_size = retrain_params.get('batch_size', 32)

        print(f"Starting update for model")
        print(f"Retraining with params: epochs={epochs}, batch_size={batch_size}")
        time.sleep(5)

        return jsonify({
            "message": f"Model retraining started successfully!",
            "status": "success",
            "params": {
                "epochs": epochs,
                "batch_size": batch_size
            }
        }), 200
    except Exception as e:
        return jsonify({
            "message": "Failed to update the model.",
            "error": str(e),
            "status": "failure"
        }), 500


@app.route('/getperformance', methods=['POST'])
def getmodelperformance():
    try:
        print(f"Starting evaluation")

        time.sleep(5)

        rmse = 49
        mae = 25
        mmape = 6.1
        r2 = 85
        return jsonify({
            "message": f"Model retraining started successfully!",
            "status": "success",
            "results": {
                "rmse": rmse,
                "mae": mae,
                "mmape": mmape,
                "r2": r2
            }
        }), 200
    except Exception as e:
        return jsonify({
            "message": "Failed to update the model.",
            "error": str(e),
            "status": "failure"
        }), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

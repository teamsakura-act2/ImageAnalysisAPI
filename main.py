import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from PIL import Image
import io
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from typing import List
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()


# CORSを回避するために追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

REFERENCE_VALUE = 20
IMG_SIZE = (100, 100)
detector = cv2.AKAZE_create()

def formatImage(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def calculateResult(target_des, comp_des, bf):
    if target_des is not None and comp_des is not None:
        target_des = target_des.astype(np.uint8)
        comp_des = comp_des.astype(np.uint8)
        matches = bf.match(target_des, comp_des)
        dist = [m.distance for m in matches]
        if len(dist) != 0:
            return sum(dist) / len(dist)
        else:
            return 10000
    else:
        return 10000

@app.get("/search/")
async def search_countryflag(base64data: str):
    
    result_code_list = []
    result_name_list = []

    # 対象画像 #########################
    target_image = Image.open(io.BytesIO(base64.b64decode(base64data)))
    target_image = formatImage(target_image)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    (target_kp, target_des) = detector.detectAndCompute(target_image, None)

    # DB接続 #########################
    uri = "mongodb+srv://sakura1234:sakura1234@cluster0.2kfseyr.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")

        # 検索先 DB/COLLECTION 指定
        db = client['ocrapi_reference_data']
        collection = db['countries']
        national_flag_list = collection.find(
            {'Country Code 2': {'$exists': True}, 'Japanese Name': {'$exists': True}, 'Base64 Data': {'$exists': True}})

        for comp in national_flag_list:

            result = 10000
            comp_code = comp['Country Code 2']
            comp_name = comp['Japanese Name']
            comp_base64 = comp['Base64 Data']

            try:
                comp_image = Image.open(io.BytesIO(base64.b64decode(comp_base64))).convert('RGB')
                comp_image = formatImage(comp_image)
                (comp_kp, comp_des) = detector.detectAndCompute(comp_image, None)
                result = calculateResult(target_des, comp_des, bf)

            except Exception as e:
                print('エラー : ' & comp_code)
                continue

            if result < REFERENCE_VALUE:
                result_code_list.append(comp_code)
                result_name_list.append(comp_name)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        # 接続のクローズ
        client.close()

    # status【'0':OK '1':NG】
    if len(result_code_list) == 1 and len(result_name_list) == 1:
        return {"code": result_code_list[0], "name": result_name_list[0], "status": "0", "errorMessage": None}
    elif len(result_code_list) > 1 and len(result_name_list) > 1:
        return {"code": None, "name": None, "status": "1", "errorMessage": "該当する国旗が2件以上存在します。"}
    else:
        return {"code": None, "name": None, "status": "1", "errorMessage": "該当する国旗がありません。"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

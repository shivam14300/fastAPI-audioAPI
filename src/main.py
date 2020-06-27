from fastapi import FastAPI,File,UploadFile,APIRouter,Form
import uvicorn
from pydantic import BaseModel
import base64
from disease_prediction import pred_lung_health


app = FastAPI(debug=True)
class Item(BaseModel):
    height: str = Form(...)
    gender: str = Form(...)
    age: str = Form(...)
    audio1: str = Form(...)
    audio2: str = Form(...)
    audio3: str = Form(...)


@app.get("/health")
async def read_root():
    return {"status": "ok"}

@app.post("/post/{item_id}")
async def update_item(item_id: int,file: Item):
    wav_file1 = open('./../data/temp1.wav','wb')
    wav_file2 = open('./../data/temp2.wav','wb')
    wav_file3 = open('./../data/temp3.wav','wb')
    file1 = base64.b64decode(file.audio1)
    file2 = base64.b64decode(file.audio2)
    file3 = base64.b64decode(file.audio3)
    wav_file1.write(file1)
    wav_file2.write(file2)
    wav_file3.write(file3)
    wav_file1.close()
    wav_file2.close()
    wav_file3.close()
    # Send the complete path of the wav files
    pred_condition1 = pred_lung_health("/home/bansalji/fastAPI-audioAPI/data/temp1.wav")
    pred_condition2 = pred_lung_health("/home/bansalji/fastAPI-audioAPI/data/temp2.wav")
    pred_condition3 = pred_lung_health("/home/bansalji/fastAPI-audioAPI/data/temp3.wav")

    return {
        "height" : file.height,
        "age" : file.age,
        "gender" : file.gender,
        "Result" : "You have XXX disease",
        "audio1_length" : pred_condition1,
        "audio2_length" : pred_condition2,
        "audio3_length" : pred_condition3,
    }


# @app.post("/try_post")
# def update_item(file: UploadFile = File(...)):
#     return {"len1": file}


# if __name__ == '__main__':
#     uvicorn.run(app,host="127.0.0.1",port="8000")
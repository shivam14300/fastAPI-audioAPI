from fastapi import FastAPI,APIRouter,File,UploadFile,Form
import uvicorn
from pydantic import BaseModel
from typing import List
import base64
app = FastAPI(debug=True)
class Item(BaseModel):
    name: str = Form(...)
    description: str = Form(...)
    price: float = Form(...)
    tax: float = Form(...)
    audio1: str = Form(...)
    audio2: str = Form(...)
    audio3: str = Form(...)

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None,p: int = 0):
    return {"item_id": item_id, "q": q, "p": p}

@app.post("/post/{item_id}")
def update_item(item_id: int,file: Item):
    print(file.audio1)
    print(file.audio2)
    print(file.audio3)
    wav_file1 = open('temp1.wav','wb')
    wav_file2 = open('temp2.wav','wb')
    wav_file3 = open('temp3.wav','wb')
    file1 = base64.b64decode(file.audio1)
    file2 = base64.b64decode(file.audio2)
    file3 = base64.b64decode(file.audio3)
    wav_file1.write(file1)
    wav_file2.write(file2)
    wav_file3.write(file3)
    return {"len1": len(file1),"len2": len(file2),"len3": len(file3) }


if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port="8000")
import base64
import logging
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
import utils
from imageio import imread
import cv2
import io

app = FastAPI()

def process_original_im(im,im_size=768):
    im_h,im_w = im.shape[:2]
    if max(im_h,im_w) < im_size or min(im_h,im_w):
        if im_w >= im_h:
            im_rh = im_size
            im_rw = int(im_w/im_h * im_size)
        elif im_w < im_h:
            im_rw = im_size
            im_rh = int(im_h/im_w * im_size)
    else:
        im_rh = im_h
        im_rw = im_w
    
    im = cv2.resize(im,(im_rw,im_rh),cv2.INTER_AREA)
    return im

def imgread(*argv):
    argv_transform = []
    for item in argv:
        item = imread(io.BytesIO(base64.b64decode(item)))
        if item.shape[2] == 4:
            item = item[:,:,0:3]
        argv_transform.append(item)
    return argv_transform

def colorbg(img,matte,bg_color:tuple):
    bg = np.full(img.shape,bg_color,np.uint8)
    res = matte * img + (1-matte) * bg
    res = np.uint8(res)
    res = utils.encode_array_to_base64(res)
    return res

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("pic-ai_6.html", context={"request": request})



################ FACE_GFPGAN E1 ####################
@app.post("/enhance1",response_class=HTMLResponse)
async def enhancer_1(request:Request,file:dict):
    try:
        file = file["img"]
        ''' content = file.split(";")[1]
        img_encoded = content.split(",")[1] '''
        img = imread(io.BytesIO(base64.b64decode(file)))
        if img.shape[2] == 4:
            img = img[:,:,0:3]
        img_pil = utils.enhance_filt(img,"f_gfpgan")
        output_img = utils.encode_array_to_base64(img_pil)
        return JSONResponse(
            status_code=200,
            content={
                "enhanced_img":output_img
            }
        )
    
    except Exception as ex:
        logging.info(ex)
        return JSONResponse(status_code=400,content={"error":str(ex)})

################ REAL_ESRGAN_FULL E2 ####################
@app.post("/enhance2",response_class=HTMLResponse)
async def enhancer_2(request:Request,file:dict):
    try:
        file = file["img"]
        img = imread(io.BytesIO(base64.b64decode(file)))
        if img.shape[2] == 4:
            img = img[:,:,0:3]
        img_pil = utils.enhance_filt(img,"realesr_full")
        output_img = utils.encode_array_to_base64(img_pil)
        return JSONResponse(
            status_code=200,
            content={
                "enhanced_img":output_img
            }
        )
    
    except Exception as ex:
        print(ex)
        logging.info(ex)
        return JSONResponse(status_code=400,content={"error":str(ex)})


############## UPCUNET ##############################
@app.post("/enhance3",response_class=HTMLResponse)
async def enhancer_3(request:Request,file:dict):
    try:
        file = file["img"]
        img = imread(io.BytesIO(base64.b64decode(file)))

        if img.shape[2] == 4:
            img = img[:,:,0:3]
        
        img_pil = utils.enhance_filt(img,"fbcnn")
        output_img = utils.encode_array_to_base64(img_pil)
        return JSONResponse(
            status_code=200,
            content = {
                "enhanced_img":output_img
            }
        )
    except Exception as ex:
        print(ex)
        logging.info(ex)
        return JSONResponse(status_code=400,content={"error":str(ex)})


############### ARCANE ###############################
@app.post("/arcane",response_class=HTMLResponse)
async def arcane_gan(request:Request,file:dict):
    try:
        file = file["img"]
        content = file.split(";")[1]
        img_encoded = content.split(",")[1]
        img = imread(io.BytesIO(base64.b64decode(img_encoded)))

        if img.shape[2] == 4:
            img = img[:,:,0:3]
        
        img_pil = utils.arcane_filt(img)
        output_img = utils.encode_array_to_base64(img_pil)
        return JSONResponse(
            status_code=200,
            content={
                'bg_img':output_img
            }
        )

    except Exception as ex:
        print(ex)
        logging.info(ex)
        return JSONResponse(status_code=400,content={"error":str(ex)})


############### ANIME ###############################
@app.post("/por2anime",response_class=HTMLResponse)
async def anime_gan(request:Request,file:dict):
    try:
        file = file["img"]
        content = file.split(";")[1]
        img_encoded = content.split(",")[1]
        img = imread(io.BytesIO(base64.b64decode(img_encoded)))

        if img.shape[2] == 4:
            img = img[:,:,0:3]
        
        img_pil = utils.anime_filt(img)
        output_img = utils.encode_array_to_base64(img_pil)
        return JSONResponse(
            status_code=200,
            content={
                'bg_img':output_img
            }
        )

    except Exception as ex:
        print(ex)
        logging.info(ex)
        return JSONResponse(status_code=400,content={"error":str(ex)})



################ COLOR-SPASH ############################
@app.post("/color_splash", response_class=HTMLResponse)
async def color_splash(request: Request, file:dict):
    try:
        file = file['img']
        content = file.split(';')[1]
        img_encoded = content.split(',')[1]
        img = imread(io.BytesIO(base64.b64decode(img_encoded)))
        if img.shape[2] == 4:
            img = img[:,:,0:3]
        img_pil = utils.bg_grayscale(img)
        output_image = utils.encode_array_to_base64(img_pil)
        return JSONResponse(
            status_code=200,
            content={
                'bg_grayscale': output_image,
            },
        )
    except Exception as ex:
        print(ex)
        logging.info(ex)
        print(ex)
        return JSONResponse(status_code=400, content={"error": str(ex)})


############## BG-REMOVER ###############################
@app.post("/", response_class=HTMLResponse)
async def remove_bg(request: Request, file:dict):
    try:
        file = file['img']
        content = file.split(';')[1]
        img_encoded = content.split(',')[1]
        img = imread(io.BytesIO(base64.b64decode(img_encoded)))
        if img.shape[2] == 4:
            img = img[:,:,0:3]
        img_pil = utils.bg_remove(img)
        output_image = utils.encode_array_to_base64(img_pil)
        return JSONResponse(
            status_code=200,
            content={
                'img_with_bk': output_image,
            },
        )
    except Exception as ex:
        logging.info(ex)
        print(ex)
        return JSONResponse(status_code=400, content={"error": str(ex)})

if __name__ == '__main__':
    uvicorn.run(app,port=9000)
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import numpy as np
import cv2
from rembg import new_session
from skimage import feature, filters
from PIL import Image
from rembg.bg import fix_image_orientation, naive_cutout, get_concat_v_multi
import os
import tempfile

app = FastAPI()

model_dir = 'u2net_human_seg'
session = new_session(model_dir)

def extract_edges(data, session):
    img = Image.fromarray(data)
    img = fix_image_orientation(img)
    masks = session.predict(img)[:1]

    cutouts = []
    for mask in masks:
        cutout = naive_cutout(img, mask)
        cutouts.append(cutout)
        break
    
    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)
        mask = get_concat_v_multi(masks)
    
    return np.asarray(cutout), np.asarray(mask)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(img_bytes)
        temp_file_path = temp_file.name
    img = cv2.imread(temp_file_path)
    os.remove(temp_file_path)

    output, mask = extract_edges(img, session=session)
    
    edges = feature.canny(mask, sigma=5)
    edges = filters.gaussian(edges, sigma=0.3)
    edges = (edges * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    dilated_edges_rgba = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2RGBA)

    alpha_factor = 0.7
    dilated_edges_rgba[:, :, 3] = (dilated_edges * alpha_factor).astype(np.uint8)
    #dilated_edges_rgba[:, :, 3] = dilated_edges

    output_with_edges = cv2.addWeighted(output, 1.0, dilated_edges_rgba, 1.0, 0)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as dilated_tempfile:
        cv2.imwrite(dilated_tempfile.name, dilated_edges_rgba)
        dilated_tempfile_path = dilated_tempfile.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_tempfile:
        cv2.imwrite(output_tempfile.name, output_with_edges)
        output_tempfile_path = output_tempfile.name

    return {
        "edges": os.path.basename(dilated_tempfile_path),
        "result": os.path.basename(output_tempfile_path)
    }

def remove_file(file_path: str):
    os.remove(file_path)

@app.get("/images/{filename}")
async def get_image(filename: str, background_tasks: BackgroundTasks):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    background_tasks.add_task(remove_file, file_path)
    return FileResponse(file_path, media_type="image/png")
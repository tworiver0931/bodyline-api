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
from ultralytics import YOLO

app = FastAPI()

yolo_model = YOLO('yolov8n.pt')

model_name = 'DIS'
session = new_session(model_name)


def crop_human(img, model, expand_ratio=0.2):
    results = model(img)

    human_boxes = [r for r in results[0].boxes if r.cls == 0]
    if human_boxes:
        x1, y1, x2, y2 = map(int, human_boxes[0].xyxy[0])

        # Expand bounding box size by the specified ratio
        width_expansion = int((x2 - x1) * expand_ratio)
        height_expansion = int((y2 - y1) * expand_ratio)

        # Calculate new coordinates with expansion
        x1 = max(0, x1 - width_expansion)
        y1 = max(0, y1 - height_expansion)
        x2 = min(img.shape[1], x2 + width_expansion)
        y2 = min(img.shape[0], y2 + height_expansion)

        cropped_img = img[y1:y2, x1:x2]
        return cropped_img, (x1, y1, x2, y2)
    else:
        return None, None
    

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

def paste_on_original_size(cropped_img, orig_size, bbox, is_edge=False):
    # Create a transparent RGBA image with the original image size
    full_img = np.zeros((orig_size[0], orig_size[1], 4), dtype=np.uint8)
    
    # Get bounding box coordinates
    x1, y1, x2, y2 = bbox
    
    # Place the cropped image into the original size canvas
    if not is_edge:
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2RGBA)
    full_img[y1:y2, x1:x2] = cropped_img
    
    return full_img

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(img_bytes)
        temp_file_path = temp_file.name
    img = cv2.imread(temp_file_path)
    os.remove(temp_file_path)

    cropped_img, bbox = crop_human(img, yolo_model)
    if cropped_img is None:
        return {
        "success": False,
        "msg": "Human not detected",
        "edges": None,
        "result": None
    }

    output, mask = extract_edges(cropped_img, session=session)
    
    # extract edges from mask
    edges = feature.canny(mask, sigma=1)
    edges = (edges * 255).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=3)

    smoothed_edges = cv2.GaussianBlur(dilated_edges, (5, 5), 0) 

    dilated_edges_rgba = cv2.cvtColor(smoothed_edges, cv2.COLOR_GRAY2RGBA)
    alpha_factor = 0.9
    dilated_edges_rgba[:, :, 3] = (dilated_edges * alpha_factor).astype(np.uint8)

    output_with_edges = cv2.addWeighted(output, 1.0, dilated_edges_rgba, 1.0, 0)

    result_full_size = paste_on_original_size(output_with_edges, img.shape[:2], bbox)
    edges_full_size = paste_on_original_size(dilated_edges_rgba, img.shape[:2], bbox, is_edge=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as dilated_tempfile:
        cv2.imwrite(dilated_tempfile.name, edges_full_size)
        dilated_tempfile_path = dilated_tempfile.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_tempfile:
        cv2.imwrite(output_tempfile.name, result_full_size)
        output_tempfile_path = output_tempfile.name

    return {
        "success": True,
        "msg": None,
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
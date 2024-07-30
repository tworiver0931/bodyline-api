from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from rembg import new_session
from skimage import feature
from PIL import Image
from rembg.bg import fix_image_orientation, naive_cutout, get_concat_v_multi
import io
import base64

app = FastAPI()

model_dir = 'u2net_human_seg.pth'
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
    img = np.array(Image.open(io.BytesIO(img_bytes)))

    output, mask = extract_edges(img, session=session)
    
    edges = feature.canny(mask, sigma=3)
    edges = (edges * 255).astype(np.uint8)

    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    dilated_edges_rgba = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2RGBA)
    dilated_edges_rgba[:, :, 3] = dilated_edges

    output_with_edges = cv2.addWeighted(output, 1.0, dilated_edges_rgba, 1.0, 0)

    _, output_img = cv2.imencode('.png', output_with_edges)
    _, edges_img = cv2.imencode('.png', dilated_edges_rgba)

    output_base64 = base64.b64encode(output_img).decode('utf-8')
    edges_base64 = base64.b64encode(edges_img).decode('utf-8')

    response = {
        'output_with_edges': output_base64,
        'dilated_edges': edges_base64
    }

    return response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
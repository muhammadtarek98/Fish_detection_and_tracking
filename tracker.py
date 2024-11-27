import numpy as np
import cv2
import torch
import torchvision
from deep_sort_realtime.deepsort_tracker import DeepSort
from torch_model import FasterRCNN
from collections import OrderedDict
import time

def enhance_image(image:np.ndarray)->np.ndarray:
    r, g, b = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_r = clahe.apply(r)
    enhanced_g = clahe.apply(g)
    enhanced_b = clahe.apply(b)
    enhanced_image = cv2.merge([enhanced_r, enhanced_g, enhanced_b])
    return enhanced_image
def load_image(image:np.ndarray) -> torch.Tensor:
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhanced_image=enhance_image(image=image)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0,0,0],
                                                                                 std=[1,1,1])])
    image_tensor = transform(enhanced_image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def apply_NMS(predictions:list, threshold:float)->list:
    pred=[]
    for i in range(len(predictions)):
        keep = torchvision.ops.nms(boxes=predictions[i]['boxes'],scores= predictions[i]['scores'], iou_threshold=threshold)
        final_prediction = dict()
        final_prediction['boxes'] = predictions[i]['boxes'][keep]
        final_prediction['scores'] = predictions[i]['scores'][keep]
        final_prediction['labels'] = predictions[i]['labels'][keep]
        pred.append(final_prediction)
    return pred
def convert_list_dict(predictions:list)->dict:
    dp=dict()
    dp['boxes'] = []
    dp['scores'] = []
    dp['labels'] = []
    for i in range(len(predictions)):
        prediction = predictions[i]
        dp['boxes'].append(prediction['boxes'].to(torch.float32))
        dp['scores'].append(prediction['scores'].to(torch.float32))
        dp['labels'].append(prediction['labels'].to(torch.int64))
    return dp

def load_model(ckpt_path:str,device:torch.device,model:torch.nn.Module)->torch.nn.Module:
    ckpt = torch.load(f=ckpt_path,
                      map_location=device,
                      weights_only=False)
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[6:]
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    return model

def model_predictions(image:torch.Tensor,model:torch.nn.Module)->dict:
    model.eval()
    with torch.no_grad():
        predictions = model(image)
    predictions = apply_NMS(predictions, threshold=0.0)
    final_predictions = convert_list_dict(predictions)
    return final_predictions
def get_the_model(ckpt_path:str,device:torch.device)->torch.nn.Module:
    model = FasterRCNN(num_classes=2)
    model=load_model(ckpt_path=ckpt_path,
           device=device,model=model)
    model=model.to(device=device)
    return model
def load_tracker():
    tracker = DeepSort(max_age=10,
                       n_init=3,
                       nms_max_overlap=1.0,
                       max_cosine_distance=0.8,
                       nn_budget=100)
    return tracker
def process_video(input_path: str, output_path: str, model: torch.nn.Module, tracker: DeepSort, device: torch.device):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    model = model.to(device=device)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    unique_fish_ids = set()
    tracking_issues = 0
    frame_count = 0
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_tensor = load_image(image=frame).to(device=device)
        predictions = model_predictions(image=frame_tensor, model=model)
        detections = []
        pred_boxes = predictions['boxes'][0]
        pred_scores = predictions['scores'][0]
        pred_labels = predictions['labels'][0]

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            xmin, ymin, xmax, ymax = box
            label = label.detach().cpu().numpy()
            score = score.detach().cpu().numpy()
            xmin = xmin.detach().cpu().numpy()
            ymin = ymin.detach().cpu().numpy()
            xmax = xmax.detach().cpu().numpy()
            ymax = ymax.detach().cpu().numpy()
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            detections.append((bbox, label, score))

        tracks = tracker.update_tracks(raw_detections=detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            if track_id not in unique_fish_ids:
                unique_fish_ids.add(track_id)
            else:
                tracking_issues += 1
            ltrb = track.to_ltrb()
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_count += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    end_time = time.time()
    processing_time = end_time - start_time
    fps = frame_count / processing_time
    summary = {
        "total_unique_fish": len(unique_fish_ids),
        "tracking_issues": tracking_issues,
        "fps": fps
    }
    return summary
"""
video_path="/home/muhammad/projects/SEE_assessment/underwater.mp4"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_the_model(ckpt_path="/home/muhammad/projects/SEE_assessment/detection_model.ckpt",
                      device=device)
tracker = load_tracker()
output_path = "/home/muhammad/projects/SEE_assessment/output_video.mp4"

process_video(input_path=video_path,model=model,output_path=output_path,tracker=tracker,device=device)
"""
#final_predictions = model_predictions(image=image_tensor,model=model)

#pred_boxes =final_predictions['boxes'][0]
#pred_labels = final_predictions['labels'][0]
##drawn_boxes = torchvision.utils.draw_bounding_boxes(
##    image=image_tensor.squeeze(),
##    boxes=pred_boxes,
#    colors="red",
#    width=2)


# filename: inference.py

import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import io

# --- CONFIGURATION (from your original project) ---
NUM_CLASSES = 4  # 3 drone classes + 1 background
IMG_SIZE = 600
PREDICTION_SCORE_THRESHOLD = 0.5
DEVICE = torch.device('cpu')
CLASS_NAMES = ['background', 'drone', 'small_drone', 'large_drone']
MODEL_PATH = "fasterrcnn_drone_detector.pth"
ID_TO_CAT_MAP = {1: 'drone', 2: 'small_drone', 3: 'large_drone'}

# --- MODEL CREATION (from your original inference.py) ---
# This function is the exact blueprint for your saved model, ensuring a perfect match.
def create_detection_model(num_classes: int) -> FasterRCNN:
    """Creates the same Faster R-CNN model structure used for training."""
    backbone = resnet18(weights=None) # Start with an un-trained backbone
    
    # This custom FPN structure is specific to your model
    return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
    in_channels_list = [128, 256, 512]
    out_channels = 256
    backbone_fpn = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )

    model = FasterRCNN(
        backbone_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )
    return model

def get_inference_transform() -> A.Compose:
    """Returns the transformation pipeline for inference."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        ToTensorV2()
    ])

def load_model():
    """Loads the model structure and the trained weights."""
    model = create_detection_model(num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        # We re-raise the error to provide the full traceback for debugging
        raise RuntimeError(f"Error loading model state_dict: {e}")

# --- PREDICTION FUNCTION (API-focused) ---
def get_drone_predictions(image_bytes: bytes, model: torch.nn.Module) -> list:
    """
    Takes an image, runs prediction, and returns results as a list of dictionaries.
    """
    transform = get_inference_transform()
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].float() / 255.0 # Normalize to [0,1]
    image_tensor = image_tensor.to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    results = []
    for box, label_id, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
        if score > PREDICTION_SCORE_THRESHOLD:
            results.append({
                "box": [int(coord) for coord in box.tolist()],
                "label": ID_TO_CAT_MAP.get(label_id.item(), 'Unknown'),
                "score": round(float(score), 4) # Round for cleaner JSON
            })
            
    return results
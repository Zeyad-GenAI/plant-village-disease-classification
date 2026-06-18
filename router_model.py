import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# القائمة الكاملة بـ 46 فئة (38 نبات + 8 صور طبيعية)
CLASSES = [
    # 38 Plant Classes (Alphabetical order as ImageFolder does)
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 
    'Soybean___healthy', 
    'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
    
    # 8 Natural Images Classes (Alphabetical order)
    'airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person'
]

class RouterModel:
    def __init__(self, model_path='combined_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = models.mobilenet_v2(weights=None)
        # الآن الطبقة الأخيرة مهيأة لـ 46 فئة
        self.model.classifier[1] = nn.Linear(self.model.last_channel, len(CLASSES))
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_t)
            _, preds = torch.max(outputs, 1)
            
        return CLASSES[preds.item()]
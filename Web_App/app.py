from flask import Flask, request, render_template
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import joblib

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- STEP 1: Load Models ---
# Load DenseNet121 model
densenet_model_path = "C:\\Users\\ezeki\\OneDrive\\Desktop\\MTechProject\\Web_App\\models\\best_densenet121_model.pth"
densenet_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, 2)
densenet_model.load_state_dict(torch.load(densenet_model_path, map_location=device))
densenet_model.to(device).eval()

# Load ResNet50 with Dropout
resnet_model_path = "C:\\Users\\ezeki\\OneDrive\\Desktop\\MTechProject\\Web_App\\models\\best_resnet50_model.pth"
class ResNet50WithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ResNet50WithDropout, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.model(x)

resnet_model = torch.load(resnet_model_path, map_location=device)
resnet_model.to(device).eval()

# Load Vision Transformer with Dropout
vit_model_path = "C:\\Users\\ezeki\\OneDrive\\Desktop\\MTechProject\\Web_App\\models\\best_vit_model.pth"
class ViTWithDropout(nn.Module):
    def __init__(self, base_model, dropout_rate=0.5):
        super(ViTWithDropout, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(base_model.embed_dim, 2)

    def forward(self, x):
        x = self.base_model.forward_features(x)
        cls_token = x[:, 0, :]
        cls_token = self.dropout(cls_token)
        x = self.classifier(cls_token)
        return x

vit_model = torch.load(vit_model_path, map_location=device)
vit_model.to(device).eval()

# Load Meta-Model (Logistic Regression)
meta_model = joblib.load("C:\\Users\\ezeki\\OneDrive\\Desktop\\MTechProject\\Web_App\\models\\meta_model.pkl")

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Label mapping
label_mapping = {0: 'The uploaded image indicates that the eye is healthy', 1: 'The uploaded image suggests signs of glaucoma. Please consult an eye care professional for a detailed examination and early treatment options.'}

# --- STEP 2: Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return render_template('predict.html', error="No file uploaded!")

        file = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        try:
            # Preprocess the image
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            # Get predictions from base models
            with torch.no_grad():
                densenet_preds = F.softmax(densenet_model(image), dim=1).cpu().numpy()
                resnet_preds = F.softmax(resnet_model(image), dim=1).cpu().numpy()
                vit_preds = F.softmax(vit_model(image), dim=1).cpu().numpy()

            # Combine predictions and pass to meta-model
            combined_preds = np.hstack((densenet_preds, resnet_preds, vit_preds))
            final_pred = meta_model.predict(combined_preds)
            predicted_label = label_mapping[int(final_pred[0])]

        except Exception as e:
            return render_template('index.html', error=f"Error during prediction: {str(e)}")

        finally:
            # Clean up uploaded file
            if os.path.exists(image_path):
                os.remove(image_path)

        # Return result
        return render_template('predict.html', prediction=predicted_label)

    return render_template('predict.html')

# --- Add Routes for Model Pages ---
@app.route('/ViT')
def vit_page():
    return render_template('ViT.html') 

@app.route('/Resnet50')
def resnet50_page():
    return render_template('Resnet50.html')  

@app.route('/DenseNet121')
def densenet121_page():
    return render_template('DenseNet121.html')  


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)

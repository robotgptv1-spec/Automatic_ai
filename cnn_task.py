import io
import uuid
import zipfile
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

class SimpleCnn(nn.Module):
    def __init__(self, num_classes=10):
        # BUG FIX: super().__init__() ya super(SimpleCnn, self).__init__() use karein
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Input size 32x32 -> MaxPool 2 baar -> 8x8 image size remaining
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class AutoImageSession:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.task_type = "image_classification"
        self.model = None
        self.class_names = []
        self.num_classes = None
        self.problem_mode = None
        self.train_log = []
        self.final_metrics = {}
        
        # Standard transforms for CNN
        self.image_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_image_zip(self, file_bytes):
        """Extracts zip file to a temp dir and creates an ImageFolder dataset"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Zip extract karna
        with zipfile.ZipFile(file_bytes, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
        
        try:
            self.image_dataset = ImageFolder(root=self.temp_dir, transform=self.image_transforms)
            self.class_names = self.image_dataset.classes
            self.num_classes = len(self.class_names)
            self.problem_mode = "multiclass" if self.num_classes > 2 else "binary"
            
            return {
                "session_id": self.id,
                "task_type": self.task_type,
                "n_samples": len(self.image_dataset),
                "classes": self.class_names
            }
        except Exception as e:
            raise ValueError(f"Invalid image folder structure inside zip: {e}")

    def train(self, epochs=10, lr=0.001, batch_size=32, test_size=0.2):
        """Trains the CNN on the extracted dataset and maintains compatibility with main.js console"""
        if not hasattr(self, 'image_dataset'):
            raise ValueError("No dataset loaded. Call load_image_zip first.")

        # Train/Test Split
        total_size = len(self.image_dataset)
        val_size = int(total_size * test_size)
        train_size = total_size - val_size
        
        train_ds, val_ds = random_split(self.image_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Initialize model with correct number of classes
        self.model = SimpleCnn(num_classes=self.num_classes)
        
        # Use CrossEntropyLoss (Binary logic handled via logits safely)[cite: 1]
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.train_log = []
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_batches = 0
            
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                n_batches += 1
            
            train_loss = running_loss / max(n_batches, 1)
            
            # Validation Step
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            val_batches = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_batches += 1
            
            test_loss = val_loss / max(val_batches, 1)
            accuracy = correct / max(total, 1)
            
            # Formatted exactly like ml_engine.py so frontend console matches perfectly![cite: 1]
            self.train_log.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "test_loss": round(test_loss, 5),
                "metric_name": "accuracy",
                "metric_value": round(accuracy, 4),
            })
            
        self.final_metrics = self.train_log[-1] if self.train_log else {}
        return {"log": self.train_log, "final": self.final_metrics, "problem_mode": self.problem_mode}

    def predict_one(self, image_bytes):
        """Predicts the class of a single uploaded image stream"""
        if self.model is None:
            raise ValueError("Train a model before predicting.")
            
        # Convert raw bytes back into an RGB PIL Image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_tensor = self.image_transforms(img).unsqueeze(0) # Batch dimension add karna (1, 3, 32, 32)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            # Match layout expected by main.js prediction handler
            probabilities = F.softmax(outputs, dim=1).numpy().ravel()
            
        pred_idx = int(np.argmax(probabilities))
        label = self.class_names[pred_idx]
        
        return {
            "prediction": str(label),
            "confidence": round(float(probabilities[pred_idx]), 4),
            "probabilities": {self.class_names[i]: round(float(p), 4) for i, p in enumerate(probabilities)},
        }
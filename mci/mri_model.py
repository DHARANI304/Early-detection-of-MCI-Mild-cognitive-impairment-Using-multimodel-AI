import torch
import torch.nn as nn
from torchvision import models, transforms as T
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

class MRIConfig:
    # Data paths
    DATA_ROOT = os.path.join("data", "Combined Dataset")  # Base directory for data
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    
    # Model parameters (default / full-training)
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # Fast-train options (for quick experiments)
    FAST_TRAIN = False
    FAST_IMAGE_SIZE = 128
    FAST_BATCH_SIZE = 8
    FAST_NUM_EPOCHS = 3
    FAST_NUM_WORKERS = 0
    
    # Normalization parameters
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Runtime configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    PIN_MEMORY = True
    NUM_CLASSES = 4
    MODEL_TYPE = "vgg"  # or "densenet"
    
    # Class mapping
    CLASS_NAMES = {
        0: "Mild Demented",
        1: "Moderate Demented",
        2: "Non Demented",
        3: "Very Mild Demented"
    }
    
    # Model save paths
    VGG_MODEL_PATH = "vgg_vgg_mri_model.pth"
    DENSENET_MODEL_PATH = "densenet_vgg_mri_model.pth"

class AlzheimerMRIClassifier:
    def __init__(self, model_type="vgg", fast_train: bool = False, use_amp: bool = True):
        """Create classifier instance.

        Args:
            model_type: 'vgg' or 'densenet'
            fast_train: if True, apply fast-training defaults (smaller images, fewer epochs, freeze backbone)
            use_amp: if True and CUDA available, use mixed precision for training
        """
        self.model_type = model_type.lower()
        self.fast_train = fast_train
        self.use_amp = use_amp and (MRIConfig.DEVICE.startswith("cuda"))
        
        # Initialize model architecture
        if self.model_type == "vgg":
            self.model = models.vgg16(pretrained=True)
            # Modify the classifier for our number of classes
            self.model.classifier[6] = nn.Linear(4096, MRIConfig.NUM_CLASSES)
        elif self.model_type == "densenet":
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(1024, MRIConfig.NUM_CLASSES)
        else:
            raise ValueError("Model type must be 'vgg' or 'densenet'")
        
        self.model = self.model.to(MRIConfig.DEVICE)

        # Apply fast-train overrides if requested
        if self.fast_train or MRIConfig.FAST_TRAIN:
            self.image_size = MRIConfig.FAST_IMAGE_SIZE
            self.batch_size = MRIConfig.FAST_BATCH_SIZE
            self.num_epochs = MRIConfig.FAST_NUM_EPOCHS
            self.num_workers = MRIConfig.FAST_NUM_WORKERS
            # default: freeze backbone to speed fine-tuning
            self.freeze_backbone = True
        else:
            self.image_size = MRIConfig.IMAGE_SIZE
            self.batch_size = MRIConfig.BATCH_SIZE
            self.num_epochs = MRIConfig.NUM_EPOCHS
            self.num_workers = MRIConfig.NUM_WORKERS
            self.freeze_backbone = False

        # Define transformations (use instance image_size)
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=MRIConfig.MEAN, std=MRIConfig.STD)
        ])

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        # Only parameters that require grad will be passed to optimizer (respecting freeze)
        if self.freeze_backbone:
            # Freeze backbone parameters (everything except classifier)
            if self.model_type == "vgg":
                for param in self.model.features.parameters():
                    param.requires_grad = False
            elif self.model_type == "densenet":
                for name, param in self.model.named_parameters():
                    if not name.startswith('classifier'):
                        param.requires_grad = False

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(trainable_params, lr=MRIConfig.LEARNING_RATE, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        # Scaler for mixed-precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def load_data(self):
        """Load and prepare the dataset"""
        # Data augmentation for training
        train_transform = T.Compose([
            T.RandomResizedCrop(self.image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(MRIConfig.MEAN, MRIConfig.STD)
        ])
        
        # Just resize and normalize for validation
        val_transform = self.transform
        
        # Create datasets
        train_dataset = ImageFolder(MRIConfig.TRAIN_DIR, transform=train_transform)
        test_dataset = ImageFolder(MRIConfig.TEST_DIR, transform=val_transform)
        
        # Create dataloaders (respect instance overrides for fast mode)
        pin_memory = MRIConfig.PIN_MEMORY and (MRIConfig.DEVICE.startswith("cuda"))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )
        
        return self.train_loader, self.test_loader

    def train_model(self, return_history=False):
        """Train the model on the dataset"""
        self.model.train()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        # Load data if not already loaded
        if not hasattr(self, 'train_loader'):
            self.load_data()
            
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            # Progress logging setup
            epoch_start_time = time.time()
            total_batches = len(self.train_loader)
            processed_samples = 0
            log_interval = 10  # print progress every N batches
            
            for inputs, labels in self.train_loader:
                # batch index for logging
                if 'batch_idx' not in locals():
                    batch_idx = 0
                inputs = inputs.to(MRIConfig.DEVICE)
                labels = labels.to(MRIConfig.DEVICE)

                self.optimizer.zero_grad()

                # Mixed precision if available
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                    # scale loss and backprop
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    with torch.set_grad_enabled(True):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        loss.backward()
                        self.optimizer.step()
                
                # Metrics
                batch_samples = inputs.size(0)
                running_loss += loss.item() * batch_samples
                running_corrects += torch.sum(preds == labels.data)
                processed_samples += batch_samples

                # Periodic progress logging
                if batch_idx % log_interval == 0:
                    elapsed = time.time() - epoch_start_time
                    samples_per_sec = processed_samples / elapsed if elapsed > 0 else 0.0
                    batches_done = batch_idx + 1
                    est_remaining_samples = (len(self.train_loader.dataset) - processed_samples)
                    est_remaining_sec = est_remaining_samples / samples_per_sec if samples_per_sec > 0 else 0.0
                    print(f"  [Epoch {epoch+1}] batch {batches_done}/{total_batches} - processed {processed_samples}/{len(self.train_loader.dataset)} samples - {samples_per_sec:.1f} samp/s - ETA {est_remaining_sec:.1f}s")
                batch_idx += 1
            
            self.scheduler.step()
            
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs = inputs.to(MRIConfig.DEVICE)
                    labels = labels.to(MRIConfig.DEVICE)
                    
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    
                    val_running_loss += loss.item() * inputs.size(0)
                    val_running_corrects += torch.sum(preds == labels.data)
            
            val_epoch_loss = val_running_loss / len(self.test_loader.dataset)
            val_epoch_acc = val_running_corrects.double() / len(self.test_loader.dataset)
            
            # Update history
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())
            history['val_loss'].append(val_epoch_loss)
            history['val_acc'].append(val_epoch_acc.item())
            
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
            
            # Save if best model
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model()
        
        print(f'Best Validation Acc: {best_acc:4f}')
        self.model.load_state_dict(best_model_wts)
        
        if return_history:
            return self.model, history
        return self.model

    def evaluate(self):
        """Evaluate the model on test data"""
        self.model.eval()
        running_corrects = 0
        
        if not hasattr(self, 'test_loader'):
            _, self.test_loader = self.load_data()
        
        for inputs, labels in self.test_loader:
            inputs = inputs.to(MRIConfig.DEVICE)
            labels = labels.to(MRIConfig.DEVICE)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
        
        test_acc = running_corrects.double() / len(self.test_loader.dataset)
        print(f'Test Accuracy: {test_acc:.4f}')
        return test_acc

    def save_model(self):
        """Save the trained model"""
        save_path = MRIConfig.VGG_MODEL_PATH if self.model_type == "vgg" else MRIConfig.DENSENET_MODEL_PATH
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load(self, model_path=None):
        """Load pre-trained model weights"""
        if model_path is None:
            model_path = MRIConfig.VGG_MODEL_PATH if self.model_type == "vgg" else MRIConfig.DENSENET_MODEL_PATH
        
        self.model.load_state_dict(torch.load(model_path, map_location=MRIConfig.DEVICE))
        self.model.eval()
    
    def predict_image(self, image_path):
        """Predict Alzheimer's stage from an MRI image path"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(MRIConfig.DEVICE)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, 1)
            
            prediction = pred.item()
            confidence = confidence.item()
            
            return {
                'class_name': MRIConfig.CLASS_NAMES[prediction],
                'confidence': confidence,
                'class_index': prediction
            }
            
        except Exception as e:
            raise Exception(f"Error predicting MRI image: {str(e)}")

def train_new_model(model_type="vgg", fast_train: bool = False, use_amp: bool = True):
    """Utility function to train a new model.

    Args:
        model_type: 'vgg' or 'densenet'
        fast_train: apply fast-train settings for quicker runs
        use_amp: use mixed-precision (if available)
    """
    classifier = AlzheimerMRIClassifier(model_type, fast_train=fast_train, use_amp=use_amp)
    classifier.load_data()
    classifier.train_model()
    test_acc = classifier.evaluate()
    return classifier, test_acc

def train_smoke_test(model_type="vgg", batches: int = 10):
    """Run a very short smoke test: train for 1 epoch on a few batches to validate speed.

    Returns the time taken (seconds) and number of processed samples.
    """
    cls = AlzheimerMRIClassifier(model_type, fast_train=True, use_amp=False)
    cls.load_data()
    # run small number of batches from train_loader
    start = time.time()
    processed = 0
    cls.model.train()
    for i, (inputs, labels) in enumerate(cls.train_loader):
        if i >= batches:
            break
        inputs = inputs.to(MRIConfig.DEVICE)
        labels = labels.to(MRIConfig.DEVICE)
        cls.optimizer.zero_grad()
        outputs = cls.model(inputs)
        loss = cls.criterion(outputs, labels)
        loss.backward()
        cls.optimizer.step()
        processed += inputs.size(0)
    elapsed = time.time() - start
    return elapsed, processed
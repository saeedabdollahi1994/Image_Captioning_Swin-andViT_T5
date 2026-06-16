import torch
import os
os.environ["HF_HOME"] = r"D:\AI_Models"

class Config:
    # 📁 مسیرهای فایل و پوشه‌ها
    IMAGE_DIR = "data/Images/"
    CAPTION_FILE = "data/captions.txt"
    CHECKPOINT_DIR = "checkpoints/"
    
    # 🧠 معماری انکودرها (Vision Transformers)
    ENCODER_SWIN = "microsoft/swin-base-patch4-window7-224-in22k"
    ENCODER_VIT = "google/vit-base-patch16-224-in21k"
    SWIN_STAGES_TO_FREEZE = 1
    VIT_LAYERS_TO_FREEZE = 3
    
    # 🗣️ معماری دیکودر (Language Model)
    DECODER_NAME = "t5-small"
    T5_DROPOUT_RATE = 0.4
    
    # 🎯 پارامترهای آموزش (Training)
    TOTAL_EPOCHS = 40
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.15
    MAX_LENGTH = 50
    
    # 🔍 پارامترهای استنتاج (Inference)
    BEAM_WIDTH = 10
    
    # ⚙️ سخت‌افزار
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
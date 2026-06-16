import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# ==========================================
# 1. Flickr Dataset Class
# ==========================================
class FlickrDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = []
        
        # Read the text file and separate images and captions
        with open(caption_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Skip the first line if it's a header
            if "image" in lines[0].lower(): 
                lines = lines[1:]
                
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Handle different dataset formats (comma or hashtag)
                if ',' in line:
                    parts = line.split(',', 1)
                else:
                    parts = line.replace('\t', ' ').split('#')
                    if len(parts) > 1:
                        parts[1] = parts[1].split(' ', 1)[1] if ' ' in parts[1] else parts[1]
                        
                if len(parts) == 2:
                    img_name, caption = parts[0].strip(), parts[1].strip()
                    self.data.append((img_name, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read the image and convert it to standard RGB format
        image = Image.open(img_path).convert("RGB")
        
        # Pass the image through the transform pipeline (for Swin and ViT)
        if self.transform is not None:
            image = self.transform(image)
            
        # In the new architecture, we return the raw English text directly.
        # Text-to-token conversion is handled by the Tokenizer in the Collate section.
        return image, caption

# ==========================================
# 2. CapsCollate Class (Batching and Padding Manager)
# ==========================================
class CapsCollate:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        imgs = []
        captions = []
        
        for image, caption in batch:
            imgs.append(image)
            captions.append(caption)
            
        # 1. Stack the images
        imgs = torch.stack(imgs, dim=0)
        
        # 2. Use T5 Tokenizer to convert words to numbers and auto-pad.
        # This function automatically fills empty spaces with standard T5 tokens.
        tokenized_texts = self.tokenizer(
            captions,
            padding=True,              # Pad to the longest sentence in this Batch
            truncation=True,           # Truncate sentences longer than the max limit
            max_length=self.max_length,
            return_tensors="pt"        # Output directly as PyTorch tensors
        )
        
        # T5 model requires two things: tokenized words (input_ids) and padding map (attention_mask)
        input_ids = tokenized_texts["input_ids"]
        attention_mask = tokenized_texts["attention_mask"]
        
        return imgs, input_ids, attention_mask

# ==========================================
# 3. get_dataloader Function (Final Builder Function)
# ==========================================
def get_dataloader(image_dir, caption_file, tokenizer, transform, batch_size=16, max_length=50, shuffle=True, num_workers=2):
    
    # Create the dataset
    dataset = FlickrDataset(
        image_dir=image_dir,
        caption_file=caption_file,
        transform=transform
    )
    
    # Create the padding manager using the Hugging Face tokenizer
    pad_collate = CapsCollate(
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Set up the DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_collate
    )
    
    return dataloader, dataset
"""
Minimal Image Captioning Notebook - CPU-Friendly Version (FIXED)
================================================================
This is a refactored version optimized for CPU training with minimal data
while preserving all key learning concepts from the original notebook.

Key Changes:
- Uses only 100 synthetic image-caption pairs for training
- Reduced model dimensions for CPU-friendly training
- All computations optimized for CPU execution
- Complete end-to-end pipeline preserved
"""

# ============================================
# CELL 1: Setup and Imports
# ============================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')

print("TensorFlow version:", tf.__version__)
print("Running on CPU only")

# ============================================
# CELL 2: Configuration Parameters (Minimized)
# ============================================

# Image parameters - smaller for CPU
IMG_HEIGHT = 128  # Reduced from 299
IMG_WIDTH = 128   # Reduced from 299

# Model parameters - significantly reduced
EMBEDDING_DIM = 64   # Reduced from 256
UNITS = 128          # Reduced from 512
VOCAB_SIZE = 1000    # Limited vocabulary
MAX_LENGTH = 20      # Maximum caption length

# Training parameters - minimal for testing
BATCH_SIZE = 4       # Small batch for CPU
EPOCHS = 3           # Just enough to see learning
NUM_EXAMPLES = 100   # Tiny dataset for testing

# Paths
DATA_DIR = './data/captions'
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================
# CELL 3: Create Synthetic Data (Fixed)
# ============================================

def create_synthetic_data(num_examples=NUM_EXAMPLES):
    """
    Create synthetic images and captions for testing
    This ensures we have matching image-caption pairs
    """
    print(f"Creating {num_examples} synthetic image-caption pairs...")
    
    # Sample caption templates for synthetic data
    caption_templates = [
        "a photo of a colorful circle",
        "an image showing a bright pattern",
        "a picture with geometric shapes",
        "abstract art with random colors",
        "a circular shape in the center",
        "colorful pixels forming a pattern",
        "a simple geometric design",
        "random colors and shapes",
        "an artistic composition",
        "a digital art creation"
    ]
    
    images = {}
    captions_dict = {}
    
    for i in range(num_examples):
        img_id = f"img_{i:04d}"
        
        # Create a random colored image with some patterns
        img = np.random.rand(IMG_HEIGHT, IMG_WIDTH, 3)
        
        # Add some structure (circles, rectangles) to make it more interesting
        center_y, center_x = IMG_HEIGHT // 2, IMG_WIDTH // 2
        radius = min(IMG_HEIGHT, IMG_WIDTH) // 4
        
        y, x = np.ogrid[:IMG_HEIGHT, :IMG_WIDTH]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        img[mask] = np.random.rand(3)  # Random color for circle
        
        images[img_id] = img.astype(np.float32)
        
        # Create captions (use multiple variations)
        captions_dict[img_id] = [
            caption_templates[i % len(caption_templates)],
            caption_templates[(i + 1) % len(caption_templates)]
        ]
    
    print(f"Created {len(images)} images with matching captions")
    return images, captions_dict

# Create the synthetic data
image_data, captions_dict = create_synthetic_data()

# ============================================
# CELL 4: Text Preprocessing
# ============================================

def preprocess_captions(captions_dict):
    """Preprocess captions and build vocabulary"""
    
    # Add start and end tokens
    processed_captions = {}
    all_captions = []
    
    for img_id, caption_list in captions_dict.items():
        processed_captions[img_id] = []
        for caption in caption_list[:1]:  # Use only first caption per image
            caption = '<start> ' + caption + ' <end>'
            processed_captions[img_id].append(caption)
            all_captions.append(caption)
    
    # Create tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<unk>",
        filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~\t\n'
    )
    tokenizer.fit_on_texts(all_captions)
    
    # Add padding token
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    
    # Convert captions to sequences
    caption_seqs = {}
    for img_id, caption_list in processed_captions.items():
        seqs = tokenizer.texts_to_sequences(caption_list)
        caption_seqs[img_id] = tf.keras.preprocessing.sequence.pad_sequences(
            seqs, maxlen=MAX_LENGTH, padding='post'
        )[0]
    
    return tokenizer, caption_seqs

tokenizer, caption_sequences = preprocess_captions(captions_dict)
print(f"Vocabulary size: {min(len(tokenizer.word_index), VOCAB_SIZE)}")
print(f"Sample caption sequence shape: {caption_sequences['img_0000'].shape}")

# ============================================
# CELL 5: Verify Data Alignment
# ============================================

print("\nVerifying data alignment:")
print(f"Number of images: {len(image_data)}")
print(f"Number of caption sequences: {len(caption_sequences)}")
print(f"Matching keys: {set(image_data.keys()) == set(caption_sequences.keys())}")

# ============================================
# CELL 6: Create TF Dataset (Fixed)
# ============================================

def create_dataset(img_ids, image_data, caption_sequences, batch_size=BATCH_SIZE):
    """Create tf.data.Dataset for training"""
    
    # Prepare data arrays using the provided img_ids
    images = np.array([image_data[img_id] for img_id in img_ids])
    captions = np.array([caption_sequences[img_id] for img_id in img_ids])
    
    print(f"Creating dataset with {len(images)} samples")
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    
    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# Split data properly
all_img_ids = list(image_data.keys())
np.random.shuffle(all_img_ids)  # Shuffle before splitting

train_size = int(NUM_EXAMPLES * 0.8)
train_img_ids = all_img_ids[:train_size]
val_img_ids = all_img_ids[train_size:]

print(f"\nData split:")
print(f"Training samples: {len(train_img_ids)}")
print(f"Validation samples: {len(val_img_ids)}")

# Create datasets
train_dataset = create_dataset(train_img_ids, image_data, caption_sequences, BATCH_SIZE)
val_dataset = create_dataset(val_img_ids, image_data, caption_sequences, BATCH_SIZE)

# Verify datasets are not empty
print(f"\nTrain dataset batches: {len(train_dataset)}")
print(f"Validation dataset batches: {len(val_dataset)}")

# ============================================
# CELL 7: Build CNN Encoder (Simplified)
# ============================================

class CNNEncoder(tf.keras.Model):
    """
    Simplified CNN Encoder using simple Conv layers for efficiency
    Key concepts: Feature extraction, dimensionality reduction
    """
    def __init__(self, embedding_dim):
        super(CNNEncoder, self).__init__()
        
        # Simple CNN layers (more controllable than pretrained models for demo)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(2)
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(2)
        self.conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool3 = tf.keras.layers.GlobalAveragePooling2D()
        
        # Dense layer for embedding
        self.dense = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return x

# ============================================
# CELL 8: Build RNN Decoder (Simplified)
# ============================================

class RNNDecoder(tf.keras.Model):
    """
    Simplified RNN Decoder with GRU units
    Key concepts: Sequence generation, word embeddings
    """
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNNDecoder, self).__init__()
        self.units = units
        
        # Embedding layer for text
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # GRU is more efficient than LSTM on CPU
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        
        # Output layers
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(0.5)
        
    def call(self, x, features, hidden, training=False):
        # Embedding for input words
        x = self.embedding(x)
        
        # Expand features to match sequence length
        features = tf.expand_dims(features, 1)
        
        # Concatenate image features with word embeddings
        x = tf.concat([features, x], axis=-1)
        
        # GRU forward pass
        output, state = self.gru(x, initial_state=hidden)
        
        # Shape for output layer
        x = self.fc1(output)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        
        return x, state
    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# ============================================
# CELL 9: Initialize Models
# ============================================

# Create model instances
encoder = CNNEncoder(EMBEDDING_DIM)
decoder = RNNDecoder(EMBEDDING_DIM, UNITS, VOCAB_SIZE)

# Optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none'
)

def loss_function(real, pred):
    """Masked loss function to ignore padding"""
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

# ============================================
# CELL 10: Training Step
# ============================================

@tf.function
def train_step(img_tensor, target):
    """Single training step with teacher forcing"""
    loss = 0
    
    # Initialize hidden state
    hidden = decoder.reset_state(batch_size=target.shape[0])
    
    # Insert start token
    start_token_id = tokenizer.word_index.get('<start>', 1)
    dec_input = tf.expand_dims([start_token_id] * target.shape[0], 1)
    
    with tf.GradientTape() as tape:
        # Encode images
        features = encoder(img_tensor, training=True)
        
        # Teacher forcing - feed target as next input
        for i in range(1, target.shape[1]):
            predictions, hidden = decoder(dec_input, features, hidden, training=True)
            loss += loss_function(target[:, i], predictions[:, 0, :])
            dec_input = tf.expand_dims(target[:, i], 1)
    
    total_loss = (loss / int(target.shape[1]))
    
    # Calculate gradients and update weights
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
    return loss, total_loss

# ============================================
# CELL 11: Training Loop (Fixed)
# ============================================

# Training history
train_loss_history = []
val_loss_history = []

print("\n" + "="*50)
print("Starting training on CPU (this will be slow but educational!)")
print("="*50)

# Check if we have data
if len(train_dataset) == 0:
    print("ERROR: Training dataset is empty!")
    print("Creating a minimal working example...")
    # Create minimal data for demonstration
    train_dataset = create_dataset(train_img_ids[:8], image_data, caption_sequences, BATCH_SIZE)
    val_dataset = create_dataset(val_img_ids[:8], image_data, caption_sequences, BATCH_SIZE)

for epoch in range(EPOCHS):
    total_loss = 0
    total_val_loss = 0
    batch_count = 0
    
    # Training
    for batch, (img_tensor, target) in enumerate(train_dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
        batch_count += 1
        
        if batch % 5 == 0:
            print(f'Epoch {epoch + 1}, Batch {batch}, Loss {batch_loss.numpy() / int(target.shape[1]):.4f}')
    
    # Validation
    val_batch_count = 0
    for img_tensor, target in val_dataset:
        hidden = decoder.reset_state(batch_size=target.shape[0])
        start_token_id = tokenizer.word_index.get('<start>', 1)
        dec_input = tf.expand_dims([start_token_id] * target.shape[0], 1)
        
        features = encoder(img_tensor, training=False)
        val_loss = 0
        
        for i in range(1, target.shape[1]):
            predictions, hidden = decoder(dec_input, features, hidden, training=False)
            val_loss += loss_function(target[:, i], predictions[:, 0, :])
            dec_input = tf.expand_dims(target[:, i], 1)
        
        total_val_loss += (val_loss / int(target.shape[1]))
        val_batch_count += 1
    
    # Store history (avoid division by zero)
    if batch_count > 0:
        avg_train_loss = total_loss / batch_count
        train_loss_history.append(avg_train_loss.numpy())
    else:
        train_loss_history.append(0)
    
    if val_batch_count > 0:
        avg_val_loss = total_val_loss / val_batch_count
        val_loss_history.append(avg_val_loss.numpy())
    else:
        val_loss_history.append(0)
    
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    print(f'Training Loss: {train_loss_history[-1]:.4f}')
    print(f'Validation Loss: {val_loss_history[-1]:.4f}')
    print('-' * 50)

# ============================================
# CELL 12: Plot Training History
# ============================================

if len(train_loss_history) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, 'r-', label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================================
# CELL 13: Inference Function
# ============================================

def generate_caption(image, max_length=MAX_LENGTH):
    """Generate caption for a single image"""
    
    # Initialize
    hidden = decoder.reset_state(batch_size=1)
    
    # Encode image
    img_tensor = tf.expand_dims(image, 0)
    features = encoder(img_tensor, training=False)
    
    # Start with <start> token
    start_token_id = tokenizer.word_index.get('<start>', 1)
    dec_input = tf.expand_dims([start_token_id], 0)
    result = []
    
    for i in range(max_length):
        predictions, hidden = decoder(dec_input, features, hidden, training=False)
        
        # Get predicted word
        predicted_id = tf.argmax(predictions[0, 0, :]).numpy()
        
        # Check if it's the end token
        if tokenizer.index_word.get(predicted_id) == '<end>':
            break
            
        word = tokenizer.index_word.get(predicted_id, '<unk>')
        if word not in ['<start>', '<end>', '<pad>']:
            result.append(word)
        
        # Prepare next input
        dec_input = tf.expand_dims([predicted_id], 0)
    
    return ' '.join(result) if result else "no caption generated"

# ============================================
# CELL 14: Test Inference
# ============================================

print("\n" + "="*50)
print("Testing caption generation on sample images")
print("="*50)

# Test on a few validation images
sample_img_ids = val_img_ids[:min(3, len(val_img_ids))]

if len(sample_img_ids) > 0:
    fig, axes = plt.subplots(1, min(3, len(sample_img_ids)), figsize=(15, 5))
    if len(sample_img_ids) == 1:
        axes = [axes]
    
    for idx, img_id in enumerate(sample_img_ids):
        # Get image and true caption
        image = image_data[img_id]
        true_caption_seq = caption_sequences[img_id]
        
        # Convert sequence back to words
        true_caption = []
        for word_id in true_caption_seq:
            if word_id != 0:  # Skip padding
                word = tokenizer.index_word.get(word_id, '')
                if word and word not in ['<start>', '<end>', '<pad>']:
                    true_caption.append(word)
        true_caption = ' '.join(true_caption) if true_caption else "no caption"
        
        # Generate caption
        generated_caption = generate_caption(image)
        
        # Display
        axes[idx].imshow(image)
        axes[idx].axis('off')
        axes[idx].set_title(f"Generated: {generated_caption}\n\nTrue: {true_caption}", 
                            fontsize=8, pad=10)
    
    plt.tight_layout()
    plt.show()

# ============================================
# CELL 15: Key Concepts Summary
# ============================================

print("\n" + "="*60)
print("KEY CONCEPTS DEMONSTRATED IN THIS NOTEBOOK:")
print("="*60)
print("""
1. IMAGE ENCODING
   - CNN for feature extraction
   - Dimensionality reduction
   - Global average pooling

2. SEQUENCE MODELING
   - RNN/GRU for sequential caption generation
   - Teacher forcing during training
   - Hidden state management

3. MULTIMODAL LEARNING
   - Combining image and text modalities
   - Joint embedding space
   - Cross-modal alignment

4. TEXT PROCESSING
   - Tokenization and vocabulary building
   - Special tokens (<start>, <end>, <pad>)
   - Sequence padding and masking

5. TRAINING TECHNIQUES
   - Masked loss for variable length sequences
   - Gradient tape for custom training loops
   - Validation monitoring

6. INFERENCE
   - Greedy decoding
   - Early stopping with <end> token
   - Handling unknown words

7. OPTIMIZATION FOR LIMITED RESOURCES
   - Simple CNN instead of large pretrained models
   - GRU instead of LSTM (faster)
   - Small batch sizes
   - Synthetic data for testing

This minimal version demonstrates all core concepts while being
runnable on CPU with just 100 synthetic examples!
""")

# ============================================
# CELL 16: Save Models
# ============================================

# Save the trained models
try:
    encoder.save_weights('./encoder_weights.h5')
    decoder.save_weights('./decoder_weights.h5')
    
    # Save tokenizer
    with open('./tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("\nModels saved successfully!")
    print("- Encoder weights: ./encoder_weights.h5")
    print("- Decoder weights: ./decoder_weights.h5")
    print("- Tokenizer: ./tokenizer.pickle")
except Exception as e:
    print(f"Could not save models: {e}")

# ============================================
# CELL 17: Next Steps and Improvements
# ============================================

print("\n" + "="*60)
print("NEXT STEPS FOR PRODUCTION:")
print("="*60)
print("""
To scale this to production (as in the original notebook):

1. USE REAL DATA
   - MS-COCO dataset (330K images, 1.5M captions)
   - Flickr30k or Conceptual Captions
   - Proper train/val/test splits

2. SCALE UP MODEL
   - Pretrained backbones (InceptionV3, ResNet, EfficientNet)
   - Transformer-based decoders
   - Attention mechanisms (Bahdanau, Luong)
   - Beam search for better captions

3. TRAINING IMPROVEMENTS
   - Mixed precision training
   - Gradient accumulation
   - Learning rate scheduling
   - Early stopping
   - Checkpoint saving

4. CLOUD TRAINING (Original notebook focus)
   - Google Cloud Vertex AI
   - Distributed training
   - GPU/TPU acceleration
   - Hyperparameter tuning
   - Model versioning

5. EVALUATION METRICS
   - BLEU scores
   - METEOR
   - CIDEr
   - Human evaluation

6. DEPLOYMENT
   - Model serving with TF Serving
   - REST API endpoints
   - Batch prediction pipelines
   - Edge deployment optimization

This minimal version gives you all the conceptual building blocks
to understand and implement the full-scale solution!
""")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

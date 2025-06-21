# Image Caption Generator

A deep learning-based image captioning system that automatically generates descriptive captions for images using Vision Transformer (ViT) and GPT-2 models.

## 🚀 Features

- **Automatic Image Captioning**: Generate natural language descriptions for any image
- **Batch Processing**: Process multiple images at once
- **GPU Support**: Automatic CUDA detection for faster processing
- **Pre-trained Models**: Uses state-of-the-art ViT-GPT2 model from Hugging Face
- **Easy-to-use Interface**: Simple Python class with intuitive methods

## 📋 Requirements

### Python Dependencies
```bash
pip install tensorflow
pip install torch
pip install transformers
pip install Pillow
pip install matplotlib
pip install numpy
pip install huggingface_hub
```

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)
- At least 4GB RAM

## 🛠️ Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd DL
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook "Image Caption Genarator.ipynb"
```

## 📖 Usage

### Basic Usage

```python
from image_captioner import ImageCaptioner

# Initialize the captioner
captioner = ImageCaptioner()

# Generate caption for a single image
image_path = "path/to/your/image.jpg"
caption = captioner.generate_caption(image_path)
print(f"Caption: {caption}")

# Process multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = captioner.process_batch(image_paths)

for path, caption in results.items():
    print(f"Image: {path}")
    print(f"Caption: {caption}\n")
```

### Advanced Usage

```python
# Customize generation parameters
caption = captioner.generate_caption(
    image_path="image.jpg",
    max_length=50,      # Maximum caption length
    num_beams=4         # Number of beams for beam search
)
```

## 🏗️ Architecture

The system uses a **Vision Encoder-Decoder** architecture:

- **Encoder**: Vision Transformer (ViT) - extracts visual features from images
- **Decoder**: GPT-2 Language Model - generates natural language captions
- **Model**: `nlpconnect/vit-gpt2-image-captioning` from Hugging Face

### Key Components

1. **ImageCaptioner Class**: Main interface for caption generation
2. **Feature Extraction**: Converts images to tensor representations
3. **Caption Generation**: Uses beam search for high-quality captions
4. **Batch Processing**: Efficiently handles multiple images

## 📊 Example Outputs

### Tennis Player
![Tennis Player](output_images/tennis_output.png)
**Caption**: "A tennis player in action on the court"

### Swimming
![Swimming](output_images/swimming_output.png)
**Caption**: "A person swimming in a pool"

### Kids Playing
![Kids Playing](output_images/kids_playing_output.png)
**Caption**: "Children playing together in a playground"

## ⚙️ Configuration

### Model Parameters

- **max_length**: Maximum number of tokens in generated caption (default: 30)
- **num_beams**: Number of beams for beam search (default: 4)
- **no_repeat_ngram_size**: Prevents repetition of n-grams (default: 2)

### Device Configuration

The system automatically detects and uses:
- CUDA GPU if available (faster processing)
- CPU as fallback

## 🔧 Customization

### Using Different Models

You can easily switch to other pre-trained models:

```python
# Example with different model
model_name = "microsoft/git-base"
captioner = ImageCaptioner(model_name=model_name)
```

### Custom Tokenization

```python
# Custom tokenizer settings
captioner.tokenizer.max_length = 100
captioner.tokenizer.padding_side = 'left'
```

## 📈 Performance

- **Processing Speed**: ~2-3 seconds per image on CPU, ~0.5-1 second on GPU
- **Caption Quality**: High-quality, contextually relevant descriptions
- **Memory Usage**: ~2GB RAM for model loading

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Download Issues**: Check internet connection and Hugging Face access
3. **Image Format Errors**: Ensure images are in common formats (JPEG, PNG, etc.)

### Error Handling

The system includes robust error handling:
- Invalid image paths
- Corrupted image files
- Network connectivity issues
- Memory constraints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the pre-trained models
- [Vision Transformer](https://arxiv.org/abs/2010.11929) paper authors
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) paper authors

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

**Note**: This project is for educational and research purposes. The generated captions may not always be perfect and should be reviewed for accuracy in production applications. 
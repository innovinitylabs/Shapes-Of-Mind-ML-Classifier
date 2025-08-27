# Emotion Classifier

A Python implementation of the [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) model for classifying emotions in English text.

## Features

- **7 Emotion Classification**: anger, disgust, fear, joy, neutral, sadness, surprise
- **Easy-to-use API**: Simple Python class interface
- **Batch Processing**: Efficiently process multiple texts
- **Confidence Filtering**: Filter predictions by confidence threshold
- **Comprehensive Analysis**: Get detailed emotion breakdowns
- **Interactive Mode**: Test the classifier interactively

## Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment using pyenv
pyenv virtualenv 3.11 emotion-classifier
pyenv activate emotion-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from emotion_classifier import EmotionClassifier

# Initialize the classifier
classifier = EmotionClassifier()

# Classify a single text
emotion = classifier.predict_emotion("I love this!")
print(emotion)  # "joy"

# Get all emotion scores
scores = classifier.predict("I love this!")
print(scores)
# [{'label': 'joy', 'score': 0.977}, ...]
```

### Advanced Usage

```python
# Confidence-based prediction
result = classifier.predict_with_confidence(
    "This is okay I guess", 
    confidence_threshold=0.7
)
print(result['confident'])  # True/False

# Batch processing
texts = ["I'm happy!", "This is terrible", "Wow, surprising!"]
results = classifier.batch_predict(texts)

# Comprehensive analysis
analysis = classifier.analyze_text("I'm scared!", verbose=True)
```

## Running Tests

### Automated Tests
```bash
python test_classifier.py
```

### Interactive Testing
```bash
python test_classifier.py
# Choose 'y' when prompted for interactive mode
```

### Demo
```bash
python emotion_classifier.py
```

## Model Information

- **Model**: [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- **Base Architecture**: DistilRoBERTa
- **Training Data**: 6 diverse datasets (~20k observations)
- **Emotions**: Ekman's 6 basic emotions + neutral
- **Accuracy**: 66% on evaluation set

### Emotion Categories

1. **anger** 🤬 - Feeling angry or mad
2. **disgust** 🤢 - Feeling disgusted or repulsed  
3. **fear** 😨 - Feeling scared or afraid
4. **joy** 😀 - Feeling happy or joyful
5. **neutral** 😐 - No strong emotion
6. **sadness** 😭 - Feeling sad or down
7. **surprise** 😲 - Feeling surprised or shocked

## Files

- `emotion_classifier.py` - Main classifier implementation
- `test_classifier.py` - Test script with examples
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.21+
- NumPy 1.21+

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{hartmann2022emotionenglish,
  author={Hartmann, Jochen},
  title={Emotion English DistilRoBERTa-base},
  year={2022},
  howpublished = {\url{https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/}},
}
```

## License

This implementation is provided as-is. Please refer to the original model's license on Hugging Face.

## Notes

- First run will download the model (~250MB)
- GPU acceleration available if CUDA is installed
- Model performs best on English text
- Optimized for social media and conversational text

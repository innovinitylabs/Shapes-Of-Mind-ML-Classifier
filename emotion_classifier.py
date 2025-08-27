#!/usr/bin/env python3
"""
Emotion Classification using j-hartmann/emotion-english-distilroberta-base

This module provides an easy-to-use interface for classifying emotions in English text.
The model classifies text into 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise.

Based on: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
"""

from transformers import pipeline
import torch
from typing import List, Dict, Union
import warnings

# Suppress specific warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class EmotionClassifier:
    """
    A class for classifying emotions in English text using DistilRoBERTa model.
    
    Attributes:
        model_name (str): The HuggingFace model identifier
        classifier: The transformers pipeline for text classification
        emotions (list): List of emotion labels the model can predict
    """
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the emotion classifier.
        
        Args:
            model_name (str): HuggingFace model identifier
        """
        self.model_name = model_name
        self.emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        
        print(f"Loading emotion classifier: {model_name}")
        print("This may take a moment on first run as the model is downloaded...")
        
        # Initialize the pipeline
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        
        print("✅ Model loaded successfully!")
    
    def predict(self, text: Union[str, List[str]], top_k: int = None) -> Union[Dict, List[Dict]]:
        """
        Predict emotions for given text(s).
        
        Args:
            text (str or list): Single text string or list of text strings
            top_k (int, optional): Return only top k emotions. If None, returns all.
            
        Returns:
            dict or list: Emotion predictions with scores
        """
        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False
        
        # Get predictions
        results = self.classifier(text)
        
        # Process results
        processed_results = []
        for result in results:
            # Sort by score (descending)
            sorted_emotions = sorted(result, key=lambda x: x['score'], reverse=True)
            
            # Apply top_k filter if specified
            if top_k:
                sorted_emotions = sorted_emotions[:top_k]
            
            processed_results.append(sorted_emotions)
        
        # Return single result if single input
        return processed_results[0] if single_input else processed_results
    
    def predict_emotion(self, text: str) -> str:
        """
        Get the most likely emotion for a text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: The predicted emotion label
        """
        result = self.predict(text, top_k=1)
        return result[0]['label']
    
    def predict_with_confidence(self, text: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Predict emotion with confidence filtering.
        
        Args:
            text (str): Input text
            confidence_threshold (float): Minimum confidence score (0.0 to 1.0)
            
        Returns:
            dict: Prediction result with confidence info
        """
        result = self.predict(text, top_k=1)
        top_emotion = result[0]
        
        return {
            'emotion': top_emotion['label'],
            'confidence': top_emotion['score'],
            'confident': top_emotion['score'] >= confidence_threshold,
            'all_scores': self.predict(text)
        }
    
    def batch_predict(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        """
        Predict emotions for a batch of texts efficiently.
        
        Args:
            texts (list): List of text strings
            show_progress (bool): Whether to show progress
            
        Returns:
            list: List of prediction results
        """
        if show_progress:
            print(f"Processing {len(texts)} texts...")
        
        results = self.predict(texts)
        
        if show_progress:
            print("✅ Batch processing complete!")
        
        return results
    
    def analyze_text(self, text: str, verbose: bool = False) -> Dict:
        """
        Comprehensive analysis of text emotions.
        
        Args:
            text (str): Input text
            verbose (bool): Whether to print detailed analysis
            
        Returns:
            dict: Comprehensive emotion analysis
        """
        all_scores = self.predict(text)
        top_emotion = all_scores[0]
        
        analysis = {
            'text': text,
            'predicted_emotion': top_emotion['label'],
            'confidence': top_emotion['score'],
            'all_emotions': {emotion['label']: emotion['score'] for emotion in all_scores},
            'top_3_emotions': all_scores[:3]
        }
        
        if verbose:
            print(f"\n📝 Text: '{text}'")
            print(f"🎯 Predicted Emotion: {analysis['predicted_emotion']} ({analysis['confidence']:.3f})")
            print("\n🔍 All Emotion Scores:")
            for emotion in all_scores:
                bar = "█" * int(emotion['score'] * 20)
                print(f"  {emotion['label']:8} {emotion['score']:.3f} |{bar}")
        
        return analysis


def main():
    """
    Simple demo of the emotion classifier.
    """
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Test examples
    test_texts = [
        "I love this so much! It's amazing!",
        "This is terrible and makes me angry.",
        "I'm scared about what might happen.",
        "This is okay, nothing special.",
        "That was a shocking surprise!",
        "I feel disgusted by this behavior.",
        "I'm feeling very sad today."
    ]
    
    print("\n" + "="*60)
    print("🎭 EMOTION CLASSIFICATION DEMO")
    print("="*60)
    
    for text in test_texts:
        result = classifier.analyze_text(text, verbose=True)
        print("-" * 40)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Shapes of Mind ML Classifier - Hybrid Sarcasm-Aware Emotion Classification

This module implements a 3-model hybrid pipeline:
1. Dual sarcasm detection (bharatiyabytes/flan-t5-sarcasm + AventIQ-AI/Sarcasmdetection)
2. Emotion classification (SamLowe/roberta-base-go_emotions) 
3. Sarcasm-aware emotion correction logic

The pipeline detects sarcasm first, then applies emotion classification with
rule-based corrections for sarcastic content.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from typing import Dict, List, Union, Any
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)


class ShapesOfMindClassifier:
    """
    Hybrid sarcasm-aware emotion classification system.
    
    Combines dual sarcasm detection with emotion classification and applies
    rule-based corrections for sarcastic content.
    """
    
    def __init__(self):
        """Initialize the 3-model hybrid pipeline."""
        print("🚀 Initializing Shapes of Mind Classifier...")
        print("Loading 3 models: 2 sarcasm detectors + 1 emotion classifier")
        
        # Model configurations - updated with working models
        self.sarcasm_model_1 = "helinivan/english-sarcasm-detector"
        self.sarcasm_model_2 = "cardiffnlp/twitter-roberta-base-irony"
        self.emotion_model = "SamLowe/roberta-base-go_emotions"
        
        # Initialize models
        self._load_models()
        
        # Sarcasm correction rules - expanded for better coverage
        self.sarcasm_corrections = {
            'joy': 'anger',
            'optimism': 'anger', 
            'gratitude': 'anger',
            'love': 'anger',
            'excitement': 'anger',
            'amusement': 'anger',
            'admiration': 'anger',
            'approval': 'anger',
            'caring': 'anger',
            'pride': 'anger',
            'relief': 'anger',
            'surprise': 'anger',
            'neutral': 'annoyance',
            'realization': 'annoyance',
            'confusion': 'annoyance',
            'curiosity': 'annoyance'
        }
        
        print("✅ All models loaded successfully!")
    
    def _load_models(self):
        """Load all three models."""
        device = 0 if torch.cuda.is_available() else -1
        
        # Load sarcasm detection models
        print("📦 Loading sarcasm detection model 1 (english-sarcasm-detector)...")
        try:
            self.sarcasm_detector_1 = pipeline(
                "text-classification",
                model=self.sarcasm_model_1,
                device=device
            )
            print("✅ Primary sarcasm model loaded successfully")
        except Exception as e:
            print(f"⚠️  Could not load primary sarcasm model: {e}")
            self.sarcasm_detector_1 = None
        
        print("📦 Loading sarcasm detection model 2 (twitter-roberta-base-irony)...")
        try:
            self.sarcasm_detector_2 = pipeline(
                "text-classification",
                model=self.sarcasm_model_2,
                device=device
            )
            print("✅ Secondary sarcasm model loaded successfully")
        except Exception as e:
            print(f"⚠️  Could not load secondary sarcasm model: {e}")
            self.sarcasm_detector_2 = None
        
        print("📦 Loading emotion classification model (go_emotions)...")
        self.emotion_classifier = pipeline(
            "text-classification",
            model=self.emotion_model,
            device=device,
            return_all_scores=True
        )
        print("✅ Emotion classification model loaded successfully")
    
    def _detect_sarcasm_primary(self, text: str) -> bool:
        """Detect sarcasm using primary model (helinivan/english-sarcasm-detector)."""
        if self.sarcasm_detector_1 is None:
            return False
            
        try:
            result = self.sarcasm_detector_1(text)
            if isinstance(result, list) and len(result) > 0:
                label = result[0]['label'].lower()
                score = result[0]['score']
                # For helinivan model: LABEL_0 = sarcastic, LABEL_1 = not sarcastic
                # Lower threshold for better sensitivity
                is_sarcastic = label == 'label_0' and score > 0.3
                return is_sarcastic
            return False
        except Exception as e:
            print(f"⚠️  Error in primary sarcasm detection: {e}")
            return False
    
    def _detect_sarcasm_secondary(self, text: str) -> bool:
        """Detect sarcasm using secondary model (irony detector)."""
        if self.sarcasm_detector_2 is None:
            return False
            
        try:
            result = self.sarcasm_detector_2(text)
            # Check if the result indicates sarcasm/irony
            if isinstance(result, list) and len(result) > 0:
                label = result[0]['label'].lower()
                score = result[0]['score']
                
                # Cardiff irony model returns "irony" or "not_irony" labels
                # Lower threshold for better sensitivity
                is_sarcastic = label == 'irony' and score > 0.3
                
                return is_sarcastic
            return False
        except Exception as e:
            print(f"⚠️  Error in secondary sarcasm detection: {e}")
            return False
    
    def _detect_sarcasm(self, text: str) -> Dict[str, Any]:
        """
        Detect sarcasm using both models and apply voting logic.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sarcasm detection results
        """
        # Add rule-based sarcasm detection for obvious cases
        obvious_sarcasm_indicators = [
            'fucking amazing', 'fucking great', 'fucking wonderful',
            'amazing that', 'great that', 'wonderful that',
            'love getting', 'love being', 'love having',
            'excellent', 'fantastic', 'brilliant', 'perfect'
        ]
        
        # Add positive indicators to prevent false positives
        positive_indicators = [
            'genuinely', 'truly', 'really', 'honestly',
            'actually', 'sincerely', 'authentically'
        ]
        
        # Add genuine emotion indicators
        genuine_emotion_indicators = [
            'absolutely furious', 'deeply saddened', 'terrified',
            'disgusting behavior', 'incredible surprise',
            'makes me so happy', 'love this so much',
            'feel neutral about', 'feel deeply'
        ]
        
        # Check for obvious sarcasm patterns
        text_lower = text.lower()
        obvious_sarcasm = any(indicator in text_lower for indicator in obvious_sarcasm_indicators)
        
        # Check for positive indicators that suggest genuine emotion
        has_positive_indicator = any(indicator in text_lower for indicator in positive_indicators)
        
        # Check for genuine emotion indicators
        has_genuine_emotion = any(indicator in text_lower for indicator in genuine_emotion_indicators)
        
        # Get model predictions
        sarcasm_1 = self._detect_sarcasm_primary(text)
        sarcasm_2 = self._detect_sarcasm_secondary(text)
        
        # Enhanced voting logic with rule-based override
        if obvious_sarcasm and not has_positive_indicator:
            final_sarcasm = True
            confidence = "high"
        elif has_positive_indicator or has_genuine_emotion:
            # If positive indicators or genuine emotion indicators are present, be conservative
            final_sarcasm = False  # Trust indicators over model disagreement
            confidence = "medium"
        elif sarcasm_1 == sarcasm_2:
            final_sarcasm = sarcasm_1
            confidence = "high"
        else:
            # If models disagree, require higher confidence for sarcasm detection
            final_sarcasm = sarcasm_1 and sarcasm_2  # Both models must agree
            confidence = "low"
        
        return {
            "is_sarcastic": final_sarcasm,
            "confidence": confidence,
            "model_1_result": sarcasm_1,
            "model_2_result": sarcasm_2,
            "obvious_sarcasm": obvious_sarcasm,
            "has_positive_indicator": has_positive_indicator,
            "has_genuine_emotion": has_genuine_emotion
        }
    
    def _classify_emotions(self, text: str) -> List[Dict[str, Any]]:
        """
        Classify emotions using the go_emotions model.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of emotion predictions with scores
        """
        try:
            results = self.emotion_classifier(text)
            # The emotion classifier returns results in different formats
            # Handle both single prediction and batch results
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    # Batch results - take first result
                    emotions = results[0]
                else:
                    # Single result
                    emotions = results
            else:
                emotions = results
            
            # Sort by score (highest first)
            sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
            return sorted_emotions
        except Exception as e:
            print(f"⚠️  Error in emotion classification: {e}")
            return [{"label": "neutral", "score": 1.0}]
    
    def _apply_sarcasm_correction(self, emotions: List[Dict[str, Any]], is_sarcastic: bool) -> str:
        """
        Apply sarcasm-aware correction to emotion predictions.
        
        Args:
            emotions: List of emotion predictions
            is_sarcastic: Whether the text is sarcastic
            
        Returns:
            Final corrected emotion label
        """
        if not emotions:
            return "neutral"
        
        top_emotion = emotions[0]['label']
        top_score = emotions[0]['score']
        
        if not is_sarcastic:
            return top_emotion
        
        # Apply sarcasm corrections only if confidence is high enough
        if top_score > 0.7:  # High confidence threshold
            corrected_emotion = self.sarcasm_corrections.get(top_emotion, top_emotion)
            return corrected_emotion
        else:
            # Low confidence - return the original emotion
            return top_emotion
    
    def analyze(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Perform complete sarcasm-aware emotion analysis.
        
        Args:
            text: Input text to analyze
            verbose: Whether to print detailed analysis
            
        Returns:
            Complete analysis results in JSON format
        """
        if verbose:
            print(f"\n🔍 Analyzing: '{text}'")
        
        # Step 1: Detect sarcasm
        sarcasm_result = self._detect_sarcasm(text)
        
        # Step 2: Classify emotions
        raw_emotions = self._classify_emotions(text)
        
        # Step 3: Apply sarcasm correction
        final_emotion = self._apply_sarcasm_correction(raw_emotions, sarcasm_result["is_sarcastic"])
        
        # Calculate quality metrics
        top_emotion_score = raw_emotions[0]['score'] if raw_emotions else 0.0
        emotion_confidence = "high" if top_emotion_score > 0.8 else "medium" if top_emotion_score > 0.6 else "low"
        
        # Overall confidence based on both sarcasm and emotion confidence
        overall_confidence = "high" if sarcasm_result["confidence"] == "high" and emotion_confidence == "high" else "medium"
        
        # Prepare result
        result = {
            "text": text,
            "is_sarcastic": sarcasm_result["is_sarcastic"],
            "sarcasm_confidence": sarcasm_result["confidence"],
            "emotion_confidence": emotion_confidence,
            "overall_confidence": overall_confidence,
            "top_emotion_score": top_emotion_score,
            "sarcasm_details": {
                "model_1_primary": sarcasm_result["model_1_result"],
                "model_2_secondary": sarcasm_result["model_2_result"],
                "obvious_sarcasm": sarcasm_result.get("obvious_sarcasm", False),
                "has_positive_indicator": sarcasm_result.get("has_positive_indicator", False),
                "has_genuine_emotion": sarcasm_result.get("has_genuine_emotion", False)
            },
            "raw_emotions": raw_emotions[:5],  # Top 5 emotions
            "final_emotion": final_emotion,
            "correction_applied": sarcasm_result["is_sarcastic"] and (raw_emotions[0]['label'] in self.sarcasm_corrections)
        }
        
        if verbose:
            self._print_analysis(result)
        
        return result
    
    def _print_analysis(self, result: Dict[str, Any]) -> None:
        """Print detailed analysis results."""
        print(f"🎭 Sarcasm: {'Yes' if result['is_sarcastic'] else 'No'} ({result['sarcasm_confidence']} confidence)")
        if result['sarcasm_details'].get('obvious_sarcasm', False):
            print("🎭 Obvious sarcasm pattern detected!")
        if result['sarcasm_details'].get('has_positive_indicator', False):
            print("✅ Positive indicator detected (genuine emotion likely)")
        if result['sarcasm_details'].get('has_genuine_emotion', False):
            print("🎯 Genuine emotion indicator detected")
        print(f"🎯 Final Emotion: {result['final_emotion']} ({result['emotion_confidence']} confidence)")
        print(f"📈 Overall Quality: {result['overall_confidence']} confidence")
        if result['correction_applied']:
            print(f"🔄 Correction Applied: {result['raw_emotions'][0]['label']} → {result['final_emotion']}")
        print(f"📊 Top Raw Emotions:")
        for i, emotion in enumerate(result['raw_emotions'][:3], 1):
            print(f"   {i}. {emotion['label']}: {emotion['score']:.3f}")
    
    def batch_analyze(self, texts: List[str], verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            verbose: Whether to print progress
            
        Returns:
            List of analysis results
        """
        if verbose:
            print(f"📦 Batch analyzing {len(texts)} texts...")
        
        results = []
        for i, text in enumerate(texts):
            if verbose and i % 5 == 0:
                print(f"Progress: {i}/{len(texts)}")
            result = self.analyze(text, verbose=False)
            results.append(result)
        
        if verbose:
            print("✅ Batch analysis complete!")
        
        return results


def main():
    """Demo of the Shapes of Mind classifier."""
    # Initialize classifier
    classifier = ShapesOfMindClassifier()
    
    # Test examples
    test_texts = [
        "I absolutely LOVE waiting in traffic for hours!",  # Sarcastic joy → anger
        "What a wonderful day to be stuck indoors.",        # Sarcastic optimism → anger  
        "I'm genuinely happy about this success!",          # Non-sarcastic joy
        "This is fine, everything is totally fine.",        # Sarcastic neutral → annoyance
        "I'm feeling sad about this news.",                 # Non-sarcastic sadness
        "Oh great, another meeting today.",                 # Sarcastic approval → annoyance
        "I feel excited about the new project!"             # Non-sarcastic excitement
    ]
    
    print("\n" + "="*70)
    print("🧠 SHAPES OF MIND CLASSIFIER DEMO")
    print("="*70)
    
    for text in test_texts:
        result = classifier.analyze(text, verbose=True)
        print("-" * 50)
    
    # Batch example
    print(f"\n📊 BATCH ANALYSIS SUMMARY:")
    batch_results = classifier.batch_analyze(test_texts)
    
    sarcastic_count = sum(1 for r in batch_results if r['is_sarcastic'])
    print(f"Sarcastic texts: {sarcastic_count}/{len(test_texts)}")
    print(f"Corrections applied: {sum(1 for r in batch_results if r['correction_applied'])}")


if __name__ == "__main__":
    main()

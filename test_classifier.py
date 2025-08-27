#!/usr/bin/env python3
"""
Test script for the emotion classifier.

This script provides examples of how to use the EmotionClassifier class
and demonstrates various features of the emotion classification model.
"""

from emotion_classifier import EmotionClassifier


def test_basic_usage():
    """Test basic emotion classification functionality."""
    print("🧪 Testing Basic Usage")
    print("-" * 30)
    
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Single prediction
    text = "I am so happy today!"
    emotion = classifier.predict_emotion(text)
    print(f"Text: '{text}'")
    print(f"Predicted emotion: {emotion}")
    
    # Detailed prediction
    result = classifier.predict(text)
    print(f"All scores: {result[:3]}")  # Show top 3
    print()


def test_confidence_filtering():
    """Test emotion prediction with confidence filtering."""
    print("🎯 Testing Confidence Filtering")
    print("-" * 35)
    
    classifier = EmotionClassifier()
    
    texts = [
        "I absolutely HATE this!",  # Should be high confidence
        "This is fine I guess.",    # Should be low confidence
        "AMAZING! Best day ever!"   # Should be high confidence
    ]
    
    for text in texts:
        result = classifier.predict_with_confidence(text, confidence_threshold=0.7)
        status = "✅ High confidence" if result['confident'] else "⚠️ Low confidence"
        print(f"Text: '{text}'")
        print(f"Emotion: {result['emotion']} ({result['confidence']:.3f}) {status}")
        print()


def test_batch_processing():
    """Test batch processing of multiple texts."""
    print("📦 Testing Batch Processing")
    print("-" * 30)
    
    classifier = EmotionClassifier()
    
    texts = [
        "This is wonderful!",
        "I'm feeling anxious about tomorrow.",
        "That's disgusting behavior.",
        "What a shocking turn of events!",
        "I'm neither happy nor sad.",
        "This makes me furious!"
    ]
    
    results = classifier.batch_predict(texts)
    
    for i, (text, result) in enumerate(zip(texts, results)):
        top_emotion = result[0]
        print(f"{i+1}. '{text}'")
        print(f"   → {top_emotion['label']} ({top_emotion['score']:.3f})")
    print()


def test_interactive_mode():
    """Interactive testing mode for user input."""
    print("💬 Interactive Mode")
    print("-" * 20)
    print("Enter text to classify emotions (type 'quit' to exit)")
    
    classifier = EmotionClassifier()
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            # Analyze the text
            analysis = classifier.analyze_text(user_input, verbose=True)
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Run all tests and demos."""
    print("🎭 EMOTION CLASSIFIER TEST SUITE")
    print("=" * 50)
    print()
    
    try:
        # Run basic tests
        test_basic_usage()
        test_confidence_filtering()
        test_batch_processing()
        
        # Ask if user wants interactive mode
        print("Would you like to try interactive mode? (y/n): ", end="")
        response = input().strip().lower()
        
        if response in ['y', 'yes']:
            test_interactive_mode()
        else:
            print("Tests completed! ✅")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure you have installed the requirements: pip install -r requirements.txt")


if __name__ == "__main__":
    main()

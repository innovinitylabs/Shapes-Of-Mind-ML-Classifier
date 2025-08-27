#!/usr/bin/env python3
"""
Test suite for Shapes of Mind Classifier.

This script provides comprehensive testing of the hybrid sarcasm-aware
emotion classification pipeline.
"""

from shapes_of_mind_classifier import ShapesOfMindClassifier
import json


def test_sarcasm_detection():
    """Test sarcasm detection capabilities."""
    print("🧪 Testing Sarcasm Detection")
    print("-" * 40)
    
    classifier = ShapesOfMindClassifier()
    
    sarcastic_examples = [
        "Oh great, another Monday morning!",
        "I just LOVE waiting in long lines.",
        "What a fantastic day to be stuck in traffic.",
        "Sure, because that's exactly what I needed today.",
        "Perfect! Another broken printer."
    ]
    
    non_sarcastic_examples = [
        "I'm genuinely excited about this opportunity!",
        "Today is a beautiful sunny day.",
        "I feel grateful for my family's support.",
        "This project is going really well.",
        "I'm happy to help you with this task."
    ]
    
    print("🎭 Sarcastic Examples:")
    for text in sarcastic_examples:
        result = classifier.analyze(text)
        status = "✅" if result['is_sarcastic'] else "❌"
        print(f"  {status} '{text}' → Sarcastic: {result['is_sarcastic']}")
    
    print("\n😊 Non-Sarcastic Examples:")
    for text in non_sarcastic_examples:
        result = classifier.analyze(text)
        status = "✅" if not result['is_sarcastic'] else "❌"
        print(f"  {status} '{text}' → Sarcastic: {result['is_sarcastic']}")
    
    print()


def test_emotion_classification():
    """Test emotion classification accuracy."""
    print("🎯 Testing Emotion Classification")
    print("-" * 40)
    
    classifier = ShapesOfMindClassifier()
    
    emotion_examples = [
        ("I'm absolutely furious about this situation!", "anger"),
        ("This makes me feel so happy and grateful!", "joy"),
        ("I'm terrified of what might happen next.", "fear"),
        ("This is disgusting and inappropriate behavior.", "disgust"),
        ("I feel deeply saddened by this news.", "sadness"),
        ("What an incredible surprise this is!", "surprise"),
        ("This is a neutral statement about the weather.", "neutral")
    ]
    
    for text, expected_category in emotion_examples:
        result = classifier.analyze(text)
        final_emotion = result['final_emotion']
        raw_emotion = result['raw_emotions'][0]['label']
        
        print(f"Text: '{text}'")
        print(f"  Raw: {raw_emotion} | Final: {final_emotion} | Expected: {expected_category}")
        if result['correction_applied']:
            print(f"  🔄 Sarcasm correction applied")
        print()


def test_sarcasm_correction():
    """Test sarcasm-aware emotion correction."""
    print("🔄 Testing Sarcasm Correction Logic")
    print("-" * 40)
    
    classifier = ShapesOfMindClassifier()
    
    correction_examples = [
        ("I just LOVE being stuck in traffic!", "Should correct joy→anger"),
        ("What a wonderful day to be indoors sick.", "Should correct optimism→anger"),
        ("This is totally fine, no problems here.", "Should correct neutral→annoyance"),
        ("I'm genuinely happy about this achievement!", "Should NOT correct (not sarcastic)"),
        ("Great job everyone, really excellent work!", "Depends on sarcasm detection")
    ]
    
    for text, expectation in correction_examples:
        result = classifier.analyze(text, verbose=True)
        print(f"Expectation: {expectation}")
        print("-" * 20)


def test_batch_processing():
    """Test batch processing functionality."""
    print("📦 Testing Batch Processing")
    print("-" * 40)
    
    classifier = ShapesOfMindClassifier()
    
    batch_texts = [
        "I love this new policy change!",  # Could be sarcastic
        "This weather is absolutely perfect.",  # Could be sarcastic
        "I'm excited about the weekend!",  # Likely genuine
        "Another software update, how exciting.",  # Likely sarcastic
        "I appreciate everyone's hard work.",  # Likely genuine
        "What a brilliant idea that was.",  # Could be sarcastic
    ]
    
    results = classifier.batch_analyze(batch_texts, verbose=True)
    
    print("\n📊 Batch Results Summary:")
    sarcastic_count = sum(1 for r in results if r['is_sarcastic'])
    corrections_count = sum(1 for r in results if r['correction_applied'])
    
    print(f"Total texts analyzed: {len(results)}")
    print(f"Detected as sarcastic: {sarcastic_count}")
    print(f"Corrections applied: {corrections_count}")
    
    print("\n📋 Detailed Results:")
    for i, (text, result) in enumerate(zip(batch_texts, results), 1):
        sarcasm_indicator = "🎭" if result['is_sarcastic'] else "😊"
        correction_indicator = "🔄" if result['correction_applied'] else ""
        print(f"{i}. {sarcasm_indicator} {correction_indicator} '{text}' → {result['final_emotion']}")
    print()


def test_json_output():
    """Test JSON output format."""
    print("📄 Testing JSON Output Format")
    print("-" * 40)
    
    classifier = ShapesOfMindClassifier()
    
    test_text = "Oh wonderful, another delay in the project timeline!"
    result = classifier.analyze(test_text)
    
    print("Sample JSON Output:")
    print(json.dumps(result, indent=2))
    
    # Validate JSON structure
    required_keys = ['text', 'is_sarcastic', 'raw_emotions', 'final_emotion']
    missing_keys = [key for key in required_keys if key not in result]
    
    if missing_keys:
        print(f"❌ Missing required keys: {missing_keys}")
    else:
        print("✅ All required keys present in JSON output")
    print()


def test_edge_cases():
    """Test edge cases and error handling."""
    print("🧩 Testing Edge Cases")
    print("-" * 40)
    
    classifier = ShapesOfMindClassifier()
    
    edge_cases = [
        "",  # Empty string
        "a",  # Single character
        "🎭😊🤔",  # Only emojis
        "This is a very long text that goes on and on and on with lots of words to test how the model handles longer inputs that might exceed typical token limits for some models and see if it gracefully handles the truncation or processing.",  # Very long text
        "Mix3d numb3rs @nd $ymb0l$!",  # Mixed characters
        "SHOUTING IN ALL CAPS WITH EXCITEMENT!!!",  # All caps
        "quiet lowercase whisper",  # All lowercase
    ]
    
    for text in edge_cases:
        try:
            result = classifier.analyze(text)
            status = "✅"
            output = f"→ {result['final_emotion']}"
        except Exception as e:
            status = "❌"
            output = f"Error: {str(e)}"
        
        display_text = repr(text) if text else "''"
        print(f"{status} {display_text} {output}")
    print()


def interactive_test():
    """Interactive testing mode."""
    print("💬 Interactive Testing Mode")
    print("-" * 40)
    print("Enter text to analyze (type 'quit' to exit)")
    
    classifier = ShapesOfMindClassifier()
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            result = classifier.analyze(user_input, verbose=True)
            
            # Offer JSON output
            show_json = input("Show JSON output? (y/n): ").strip().lower()
            if show_json in ['y', 'yes']:
                print("\nJSON Output:")
                print(json.dumps(result, indent=2))
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Run all tests."""
    print("🧠 SHAPES OF MIND CLASSIFIER TEST SUITE")
    print("=" * 60)
    print()
    
    try:
        # Run all automated tests
        test_sarcasm_detection()
        test_emotion_classification()
        test_sarcasm_correction()
        test_batch_processing()
        test_json_output()
        test_edge_cases()
        
        print("✅ All automated tests completed!")
        print("\nWould you like to try interactive testing? (y/n): ", end="")
        response = input().strip().lower()
        
        if response in ['y', 'yes']:
            interactive_test()
        else:
            print("Testing completed! 🎉")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure you have installed the requirements: pip install -r requirements.txt")


if __name__ == "__main__":
    main()

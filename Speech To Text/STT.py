import sounddevice as sd
import vosk
import json
import queue
import random
import time
import nltk 

def load_word_dictionary():
    """
    Loads the word list from NLTK and filters it for the game.
    """
    try:
        print("Loading NLTK word dictionary...")
        from nltk.corpus import words
        word_list = words.words()
        
        # Filter the list to make the game more enjoyable:
        # - Words between 6 and 15 letters long.
        # - All lowercase.
        # - Only contains alphabetic characters (no hyphens, apostrophes, etc.).
        filtered_words = [
            word.lower() for word in word_list 
            if 6 <= len(word) <= 15 and word.isalpha()
        ]
        
        print(f"Loaded {len(filtered_words)} suitable words for the game.")
        return filtered_words
    except LookupError:
        print("\n--- NLTK 'words' corpus not found! ---")
        print("Please run the download script first (see instructions).")
        print("You can create a file named 'download_nltk_data.py' with:")
        print("import nltk\nnltk.download('words')")
        return None
    except ImportError:
        print("\n--- NLTK library not found! ---")
        print("Please install it by running: pip install nltk")
        return None

# Load the dictionary at the start
WORD_DICTIONARY = load_word_dictionary()

# --- Vosk & Audio Configuration ---
MODEL_PATH = "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
DEVICE = None

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_queue.put(bytes(indata))

def generate_challenge():
    """
    Selects a random word from the dictionary and creates a snippet from it.
    """
    target_word = random.choice(WORD_DICTIONARY)
    
    if len(target_word) <= 5:
        snippet_length = 3
    else:
        snippet_length = random.randint(3, 5)

    max_start_index = len(target_word) - snippet_length
    start_index = random.randint(0, max_start_index)
    
    snippet = target_word[start_index : start_index + snippet_length]
    
    return target_word, snippet

def run_game():
    """Main function to run the speech recognition game."""
    # Check if the dictionary loaded successfully
    if not WORD_DICTIONARY:
        print("Exiting due to dictionary loading failure.")
        return
        
    print("\n--- Speech Recognition Game ---")
    print("Loading Vosk model...")

    try:
        model = vosk.Model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading Vosk model: {e}")
        return

    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    print("Model loaded. Get ready!")
    time.sleep(2)

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, device=DEVICE,
                           dtype='int16', channels=1, callback=audio_callback):
        
        while True:
            target_word, current_snippet = generate_challenge()
            
            print("---------------------------------")
            print(f"Say a word containing the snippet: '{current_snippet.upper()}'")
            print("Listening...")

            while True: 
                data = audio_queue.get()

                if recognizer.AcceptWaveform(data):
                    result_json = recognizer.Result()
                    result_dict = json.loads(result_json)
                    transcribed_text = result_dict.get('text', '').lower()

                    if transcribed_text:
                        print(f"🎤 You said: '{transcribed_text}'")
                        is_correct = check_answer(transcribed_text, current_snippet)
                        if is_correct:
                            print(f"The original word was: '{target_word}'")
                        break 
            
            print("\nNext round in 3 seconds...")
            time.sleep(3)

def check_answer(transcribed_word, snippet):
    """
    Compares the transcribed word with the game snippet.
    """
    if snippet in transcribed_word:
        print(f"✅ CORRECT! '{transcribed_word}' contains '{snippet}'.")
        return True
    else:
        print(f"❌ INCORRECT. Your word must contain '{snippet}'.")
        return False

if __name__ == "__main__":
    run_game()
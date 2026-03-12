import sounddevice as sd
import numpy as np
import torch
import time

# -----------------------------------------------------------------
# 1. Configuration & Thresholds
# -----------------------------------------------------------------
SAMPLE_RATE = 16000          # Required by Silero VAD
CHUNK = 512                  # Process 32ms of audio at a time

SPEECH_PROB_THRESHOLD = 0.5  # Confidence required to count as speech
CHEATING_DURATION = 5.0      # Seconds of speech required to count as 1 violation
CHEATING_THRESHOLD = 5       # Number of violations needed to flag as a cheater
ALLOWED_PAUSE = 1.5          # Seconds of silence allowed before resetting the timer

# -----------------------------------------------------------------
# 2. Load the Silero VAD Model
# -----------------------------------------------------------------
print("Loading Silero VAD model...")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
print("Model loaded successfully!")

# State tracking variables
is_speaking = False
speech_start_time = 0.0
last_speech_time = 0.0
cheat_counter = 0            # New variable to track violations

print(f"Listening... (Speak for {CHEATING_DURATION} seconds to get a strike. {CHEATING_THRESHOLD} strikes = Cheater. Press Ctrl+C to stop.)")

try:
    # -----------------------------------------------------------------
    # 3. Initialize Microphone Stream using sounddevice
    # -----------------------------------------------------------------
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            audio_chunk, overflowed = stream.read(CHUNK)
            audio_float32 = audio_chunk.flatten()
            tensor_chunk = torch.from_numpy(audio_float32)
            
            speech_prob = model(tensor_chunk, SAMPLE_RATE).item()
            current_time = time.time()
            
            # -----------------------------------------------------------------
            # 4. The Timing & Threshold Logic
            # -----------------------------------------------------------------
            if speech_prob > SPEECH_PROB_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    speech_start_time = current_time
                    print("\n[Info] Speech started. Tracking duration...")
                
                last_speech_time = current_time
                duration = current_time - speech_start_time
                
                # --- NEW: Real-time logging of timer and counter ---
                # \r overwrites the current line in the terminal so it looks like a live timer
                print(f"\r⏳ Timer: {duration:.1f}s / {CHEATING_DURATION}s | 🛑 Strikes: {cheat_counter}/{CHEATING_THRESHOLD}", end="")
                
                # Check if the speaking time has reached the 5-second duration
                if duration >= CHEATING_DURATION:
                    cheat_counter += 1
                    print(f"\n⚠️ Violation recorded! Spoke for {CHEATING_DURATION} seconds. Total Strikes: {cheat_counter}/{CHEATING_THRESHOLD}")
                    
                    if cheat_counter >= CHEATING_THRESHOLD:
                        print("\n🚨 CHEATING DETECTED: User exceeded the maximum allowed speaking limits! 🚨\n")
                        cheat_counter = 0  # Reset the counter after catching them (optional)
                    
                    # Reset the speech timer so we can start timing the *next* 5 seconds
                    # if they just keep talking without taking a break.
                    speech_start_time = current_time
                    
            else:
                if is_speaking:
                    time_since_last_word = current_time - last_speech_time
                    if time_since_last_word > ALLOWED_PAUSE:
                        print(f"\n[Info] Speech stopped or pause was too long. Timer reset. Current Strikes: {cheat_counter}")
                        is_speaking = False

except KeyboardInterrupt:
    print("\nStopping the stream...")
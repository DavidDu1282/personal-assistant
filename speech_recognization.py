#%%
import threading
import json
import time
import pyaudio
from vosk import Model, KaldiRecognizer

# Voice recognition class that continuously listens and calls a callback with recognized text
class VoskRecognizer:
    def __init__(self, model_path, callback):
        self.model = Model(model_path)  # Load the VOSK model
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.callback = callback
        self.is_listening = False

    def start_listening(self):
        self.is_listening = True
        # Start listening in a separate thread
        threading.Thread(target=self._listen_in_background, daemon=True).start()

    def stop_listening(self):
        self.is_listening = False

    def _listen_in_background(self):
        # Initialize the audio stream for VOSK
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
        stream.start_stream()
        
        print("Listening for voice commands...")

        while self.is_listening:
            data = stream.read(4096, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                command = result.get("text", "")
                if command:  # If there's recognized text
                    print(f"Recognized: {command}")
                    self.callback(command)  # Call the callback with recognized text

            # Adding a small delay to prevent rapid looping
            time.sleep(0.1)

        # Clean up the audio stream when stopping
        stream.stop_stream()
        stream.close()
        p.terminate()

# Command processor class that handles recognized text
# class CommandProcessor:
#     def process_command(self, command):
#         print(f"Processing command: {command}")
#         # Add logic here for handling different commands

# # Callback function that directs recognized text to CommandProcessor
# def command_callback(command):
#     processor.process_command(command)

# if __name__ == '__main__':
#     # Initialize the CommandProcessor
#     processor = CommandProcessor()

#     # Path to your downloaded VOSK model directory
#     model_path = Model("./vosk-model-en-us-0.22")  # Download and specify model path

#     # Instantiate VoskRecognizer with the callback function
#     recognizer = VoskRecognizer(model_path=model_path, callback=command_callback)
#     recognizer.start_listening()

#     try:
#         while True:
#             time.sleep(1)  # Keep main thread alive
#     except KeyboardInterrupt:
#         recognizer.stop_listening()
#         print("Stopped listening.")

#%%
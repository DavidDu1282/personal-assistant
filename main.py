#%%
import LLM
import query_index
import speech_recognization
import time


#%%
# Command processor class that handles recognized text
class CommandProcessor:
    def process_command(self, command):
        print(f"Processing command: {command}")

        results_df = query_index.query_index(command)
        for index, row in results_df.iterrows():
            print(row['title'], row['text'])
        summary = LLM.summarize_documents(results_df=results_df, max_length=150)
        prompt = LLM.create_prompt(query=command, summaries=summary)
        response = LLM.generate_response(prompt=prompt,max_length=500)

# Callback function that directs recognized text to CommandProcessor
def command_callback(command):
    processor.process_command(command)

def summarize_documents(results_df, max_length=150):
    """
    Summarizes documents to extract key points.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing the documents.
    - max_length (int): Maximum length for each summary.

    Returns:
    - str: A combined summary of key points from the documents.
    """

    summarizer = pipeline("summarization")
    summaries = []

    for _, row in results_df.iterrows():
        title = row['title']
        text = row['text']
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        summaries.append(f"{title}: {summary[0]['summary_text']}")

    return " ".join(summaries)

if __name__ == '__main__':
    # Initialize the CommandProcessor
    processor = CommandProcessor()

    # Path to your downloaded VOSK model directory
    model_path = "./vosk-model-en-us-0.22"  # Download and specify model path

    # Instantiate VoskRecognizer with the callback function
    recognizer = speech_recognization.VoskRecognizer(model_path=model_path, callback=command_callback)
    recognizer.start_listening()

    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        recognizer.stop_listening()
        print("Stopped listening.")

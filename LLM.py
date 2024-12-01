#%%
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#%%
def summarize_documents(results_df, model = "sshleifer/distilbart-cnn-12-6", max_length=150):
    """
    Summarizes documents to extract key points.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing the documents.
    - max_length (int): Maximum length for each summary.

    Returns:
    - str: A combined summary of key points from the documents.
    """

    summarizer = pipeline("summarization", model=model)
    summaries = []

    for _, row in results_df.iterrows():
        title = row['title']
        text = row['text']
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        summaries.append(f"{title}: {summary[0]['summary_text']}")

    return " ".join(summaries)

def create_prompt(query, summaries):
    """
    Creates a structured prompt for LLM response generation.

    Parameters:
    - query (str): The original query from the user.
    - summaries (str): Summarized content from related documents.

    Returns:
    - str: The structured prompt to use with the LLM.
    """
    prompt = f"Answer the following question based on the context provided.\n\nQuestion: {query}\n\nContext:\n{summaries}\n\nAnswer:"
    return prompt

def generate_response(prompt, 
                      #model_name="unsloth/llama-3-8b-Instruct-bnb-4bit", 
                      model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                      max_length=100):
    """
    Generates a response using an LLM based on a structured prompt.

    Parameters:
    - prompt (str): The input prompt with query and context.
    - model_name (str): The name of the LLM model to use.
    - max_length (int): The maximum length of the generated response.

    Returns:
    - str: The generated response.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt")#.to('cuda')
    attention_mask = inputs["attention_mask"]
    outputs = model.generate(inputs['input_ids'],
                             attention_mask=attention_mask,
                             pad_token_id=tokenizer.eos_token_id,
                             max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

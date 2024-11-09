from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

def generate_response(prompt, model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", max_length=100):
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
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

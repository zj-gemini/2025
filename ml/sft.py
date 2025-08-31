from huggingface_hub import login, whoami

# This line attempts to log you into the Hugging Face Hub using an access token.
# You need to get this token from your Hugging Face account settings.
# For some models like Gemma, you must first agree to their terms of use on the model's page.

# This specifies the pre-trained model we want to use from the Hugging Face Hub.
BASE_MODEL = "google/gemma-3-270m-it"

try:
    # whoami() checks if the login was successful and retrieves your user information.
    user = whoami()
    print("You are logged in as:", user["name"])
except Exception as e:
    print("You are not logged in. Error:", e)


from transformers import AutoTokenizer, AutoModelForCausalLM

# AutoTokenizer.from_pretrained downloads the tokenizer specific to the chosen model.
# The tokenizer is responsible for converting text into a sequence of numbers (tokens)
# that the model can understand.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# AutoModelForCausalLM.from_pretrained downloads the actual pre-trained model weights.
# "ForCausalLM" means this model is designed for causal language modeling
# (i.e., predicting the next word in a sequence).
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# This is the standard format for providing input to chat models.
# It's a list of dictionaries, where each dictionary has a 'role' (user or assistant) and 'content'.
messages = [
    {"role": "user", "content": "Who are you?"},
]

# This function takes our message and formats it exactly as the Gemma model expects for a chat.
# It handles adding special tokens like <start_of_turn> and <end_of_turn>.
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # Ensures the model knows it's its turn to generate a response.
    tokenize=True,  # Converts the formatted text into token IDs.
    return_dict=True,  # Returns a dictionary containing 'input_ids' and 'attention_mask'.
    return_tensors="pt",  # Returns the data as PyTorch tensors.
).to(
    model.device
)  # Moves the input tensors to the same device (CPU or GPU) as the model.

# This is the core generation step.
# `**inputs` unpacks the dictionary from the tokenizer into arguments for the generate function.
# `max_new_tokens` limits the length of the generated response.
outputs = model.generate(**inputs, max_new_tokens=40)

# The `outputs` tensor contains both the original input tokens and the newly generated ones.
# This line decodes only the newly generated part back into human-readable text
# by slicing the output tensor, starting from the end of the input tokens.
print(
    tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
)

# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

import onnxruntime_genai as og

def model_load(model_dir : str):

    model = og.Model(model_dir)
    return model

def get_tokenizer(model):
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    return tokenizer, tokenizer_stream

def get_prompt():
    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

    text = input("Input: ")
    if not text:
        print("Error, input cannot be empty")
        exit

    prompt = f'{chat_template.format(input=text)}'
    return prompt

def setup(model, tokenizer, tokenizer_stream, prompt):
    
    # Set the max length to something sensible by default,
    # since otherwise it will be set to the entire context length
    search_options = {}
    search_options['max_length'] = 512

    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    
    print("Creating generator with prompt")
    generator = og.Generator(model, params)

    print("Output: ", end='', flush=True)

    # Get the token IDs for </|assistant|> as stop sequence
    stop_sequence = "</|assistant|>"
    stop_tokens = tokenizer.encode(stop_sequence)
    print(f"\nStop sequence '{stop_sequence}' encoded as tokens: {stop_tokens}")

    return generator, stop_tokens

def run(generator, tokenizer_stream, stop_tokens):
    num_tokens = 0
    tokens = []
    
    # For detecting stop sequence
    token_buffer = []
    stop_tokens_len = len(stop_tokens)
    
    # For debugging - collect the full text
    full_text = ""
    
    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            token_text = tokenizer_stream.decode(new_token)
            
            tokens.append(new_token)
            token_buffer.append(new_token)
            
            # Keep only the most recent tokens needed for comparison
            if len(token_buffer) > stop_tokens_len:
                token_buffer.pop(0)
            
            # Print the generated token and update full text
            print(token_text, end='', flush=True)
            full_text += token_text
            num_tokens += 1
            
            # Check if we've generated the stop sequence
            if len(token_buffer) == stop_tokens_len:
                if token_buffer == stop_tokens:
                    print("\n[STOP SEQUENCE DETECTED - Ending generation]")
                    break
                
            # Alternative check: look for "</|assistant|>" in the full text
            if "</|assistant|>" in full_text:
                print("\n[DETECTED END OF RESPONSE IN TEXT - Ending generation]")
                break
                
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")

    print()
    print(f"total tokens: {num_tokens}")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model', help='Path to model directory with config json')
    args = parser.parse_args()
    print(f"model_dir: {args.model_dir}")

    model = model_load(args.model_dir)
    tokenizer, tokenizer_stream = get_tokenizer(model)
    prompt = get_prompt()
    generator, stop_tokens = setup(model, tokenizer, tokenizer_stream, prompt)
    run(generator, tokenizer_stream, stop_tokens)
    del generator

db_path = "../vectorstores/db"

model_path = "../../model/"
model_name = "llama-2-7b-chat.ggmlv3.q8_0.bin"

model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin?download=true"

prompt_template = """
<s>[INST] <<SYS>>
Insert instructions here!
<</SYS>>

{context}

{question} [/INST]
"""

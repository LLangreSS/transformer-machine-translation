
from src.data.tokenizer import Tokenizer

en_tokenizer = Tokenizer("tokenizers/en.model")
ru_tokenizer = Tokenizer("tokenizers/ru.model")

en_text = "Hello, how are you?"
ru_text = "Привет, как твои дела?"

en_ids = en_tokenizer.encode(en_text)
ru_ids = ru_tokenizer.encode(ru_text)

print("English:")
print(en_ids)
print(en_tokenizer.decode(en_ids))

print("\nRussian:")
print(ru_ids)
print(ru_tokenizer.decode(ru_ids))

print("\nVocabulary sizes:")
print(en_tokenizer.vocab_size)
print(ru_tokenizer.vocab_size)

print("\nSpecial IDs:")
print(en_tokenizer.pad_id, en_tokenizer.bos_id, en_tokenizer.eos_id)
print(ru_tokenizer.pad_id, ru_tokenizer.bos_id, ru_tokenizer.eos_id)

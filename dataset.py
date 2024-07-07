import torch
from pathlib import Path
import json
import string
import re
from collections import Counter

class WineDataset(torch.utils.data.Dataset):
    """Custom Dataset for the Wine Reviews."""
    def __init__(self, path_to_file: Path | str, max_length: int, vocab_size: int):
        """Init variables."""
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        with open(path_to_file, encoding="utf-8") as json_file:
            wine_data = json.load(json_file)

        filtered_data = [
            f"wine review : {country} : {province} : {variety} : {description}"
            for x in wine_data
            if all(
                (
                    country := x.get("country"),
                    province := x.get("province"),
                    variety := x.get("variety"),
                    description := x.get("description"),
                )
            )
        ]

        text_dataset = [self.prepare_string(x) for x in filtered_data]
        self.vocab = self.create_vocab(text_dataset)
        self.train_dataset = [self.prepare_inputs(text) for text in text_dataset]

    def prepare_string(self, text: str) -> list[str]:
        """Prepare a string."""
        # Replace punctuation characters and new lines
        # with itself surronded by spaces
        text = re.sub(f"([{string.punctuation}, '\n'])", r" \1 ", text)
        # Replace a sequence of spaces with a space
        text = re.sub(" +", " ", text)
        text = text.lower().split()
        return text
    
    def create_vocab(self, texts: list[list[str]]) -> dict[int, str]:
        """Create a vocab from the dataset."""
        counter = Counter(token for tokens in texts for token in tokens)
        counter = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
        vocab = [word for (word, _) in counter.items()]
        vocab.insert(0, '')
        vocab.insert(1, '<unk>')
        vocab = vocab[:self.vocab_size]
        vocab = {word: idx for idx, word in enumerate(vocab)}
        return vocab

    def get_vocab(self) -> dict[int, str]:
        """Return the vocab."""
        return self.vocab
    
    def prepare_inputs(self, text: str) -> list[int]:
        """Tokenize the text."""
        tokenized_text = [self.vocab.get(word, 1) for word in text]
        if len(tokenized_text) >= self.max_length:
            return tokenized_text[:self.max_length]
        len_to_pad = self.max_length - len(tokenized_text)
        tokenized_text += [0 for _ in range(len_to_pad)]
        return tokenized_text
    
    def __len__(self) -> int:
        return len(self.train_dataset)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.train_dataset[idx]
        return torch.Tensor(text[:-1]), torch.Tensor(text[1:])

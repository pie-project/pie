"""Random synthetic dataset for stress testing."""

import random
import string
from typing import List

from .base import DatasetLoader, Request


class RandomDataset(DatasetLoader):
    """Generate random prompts with configurable input/output lengths."""
    
    def __init__(
        self,
        input_len: int = 512,
        output_len: int = 128,
        range_ratio: float = 0.0,
        seed: int = 42,
    ):
        """
        Args:
            input_len: Average input prompt length in tokens (estimated as chars/4).
            output_len: Expected output length in tokens.
            range_ratio: Ratio for length variation (0.0 = fixed, 0.5 = +/-50%).
            seed: Random seed for reproducibility.
        """
        self.input_len = input_len
        self.output_len = output_len
        self.range_ratio = range_ratio
        self.seed = seed
    
    def load(self, num_requests: int) -> List[Request]:
        random.seed(self.seed)
        requests = []
        
        for _ in range(num_requests):
            # Apply range variation
            if self.range_ratio > 0:
                input_variation = int(self.input_len * self.range_ratio)
                output_variation = int(self.output_len * self.range_ratio)
                actual_input_len = random.randint(
                    max(1, self.input_len - input_variation),
                    self.input_len + input_variation
                )
                actual_output_len = random.randint(
                    max(1, self.output_len - output_variation),
                    self.output_len + output_variation
                )
            else:
                actual_input_len = self.input_len
                actual_output_len = self.output_len
            
            # Generate random text (chars ≈ tokens * 4)
            char_len = actual_input_len * 4
            prompt = self._generate_random_text(char_len)
            
            requests.append(Request(
                prompt=prompt,
                expected_output_len=actual_output_len,
            ))
        
        return requests
    
    def _generate_random_text(self, length: int) -> str:
        """Generate pseudo-random readable text."""
        words = []
        current_len = 0
        
        while current_len < length:
            word_len = random.randint(3, 10)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
            words.append(word)
            current_len += word_len + 1  # +1 for space
        
        return ' '.join(words)[:length]

"""ShareGPT dataset loader for realistic chat workloads."""

import json
import random
from pathlib import Path
from typing import List, Optional

from .base import DatasetLoader, Request


class ShareGPTDataset(DatasetLoader):
    """Load prompts from ShareGPT dataset (real ChatGPT conversations)."""
    
    HUGGINGFACE_DATASET = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    DEFAULT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
    
    def __init__(
        self,
        path: Optional[str] = None,
        max_input_len: int = 1024,
        max_output_len: int = 512,
        seed: int = 42,
    ):
        """
        Args:
            path: Path to ShareGPT JSON file. If None, downloads from HuggingFace.
            max_input_len: Maximum input length in tokens (truncates longer prompts).
            max_output_len: Maximum expected output length.
            seed: Random seed for shuffling.
        """
        self.path = path
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.seed = seed
        self._data = None
    
    def _download_dataset(self) -> Path:
        """Download ShareGPT dataset from HuggingFace."""
        from huggingface_hub import hf_hub_download
        
        cache_dir = Path.home() / ".cache" / "pie-bench" / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        local_path = cache_dir / self.DEFAULT_FILENAME
        if local_path.exists():
            return local_path
        
        print(f"Downloading ShareGPT dataset from HuggingFace...")
        downloaded = hf_hub_download(
            repo_id=self.HUGGINGFACE_DATASET,
            filename=self.DEFAULT_FILENAME,
            repo_type="dataset",
            local_dir=cache_dir,
        )
        return Path(downloaded)
    
    def _load_data(self) -> List[dict]:
        """Load and parse the ShareGPT JSON file."""
        if self._data is not None:
            return self._data
        
        if self.path:
            file_path = Path(self.path)
        else:
            file_path = self._download_dataset()
        
        print(f"Loading ShareGPT data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        
        return self._data
    
    def load(self, num_requests: int) -> List[Request]:
        data = self._load_data()
        random.seed(self.seed)
        random.shuffle(data)
        
        requests = []
        
        for conversation in data:
            if len(requests) >= num_requests:
                break
            
            # Extract the first human message as prompt
            conversations = conversation.get("conversations", [])
            if not conversations:
                continue
            
            # Find first human turn
            prompt = None
            expected_output_len = self.max_output_len
            
            for i, turn in enumerate(conversations):
                role = turn.get("from", "")
                value = turn.get("value", "")
                
                if role == "human" and value.strip():
                    prompt = value.strip()
                    
                    # If there's a following assistant response, use its length
                    if i + 1 < len(conversations):
                        next_turn = conversations[i + 1]
                        if next_turn.get("from") == "gpt":
                            # Estimate tokens as chars / 4
                            expected_output_len = min(
                                len(next_turn.get("value", "")) // 4,
                                self.max_output_len
                            )
                    break
            
            if not prompt:
                continue
            
            # Truncate if needed (chars ≈ tokens * 4)
            max_chars = self.max_input_len * 4
            if len(prompt) > max_chars:
                prompt = prompt[:max_chars]
            
            requests.append(Request(
                prompt=prompt,
                expected_output_len=max(1, expected_output_len),
            ))
        
        if len(requests) < num_requests:
            print(f"Warning: Only found {len(requests)} valid prompts (requested {num_requests})")
        
        return requests

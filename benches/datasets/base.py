"""Base classes for dataset loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class Request:
    """A single benchmark request."""
    prompt: str
    expected_output_len: int  # For throughput calculation


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load(self, num_requests: int) -> List[Request]:
        """Load requests for benchmarking.
        
        Args:
            num_requests: Number of requests to generate/load.
            
        Returns:
            List of Request objects.
        """
        pass

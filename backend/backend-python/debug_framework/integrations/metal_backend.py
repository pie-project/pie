"""
Metal Backend Implementation

Provides real Metal compute integration for tensor computation
using the Metal backend and Python bindings.
"""

import os
import sys
import time
import warnings
import subprocess
from typing import Dict, Any, Optional
import numpy as np

from .backend_interfaces import BackendInterface, BackendType, TensorComputationResult


class MetalBackend(BackendInterface):
    """
    Metal backend using actual Metal compute kernels.

    This backend provides access to optimized Metal compute kernels for
    tensor operations, executing on Apple's Metal Performance Shaders.
    """

    def __init__(self, metal_backend_path: Optional[str] = None, model_metadata: Optional[Dict[str, Any]] = None):
        super().__init__(BackendType.METAL)
        self.metal_backend_path = metal_backend_path
        self._metal_executor = None
        self._metallib_path: Optional[str] = None
        self._available_kernels: Dict[str, bool] = {}
        self._device_info: Optional[str] = None

        # Store model metadata for attention parameters
        self._model_metadata = model_metadata or {}
        self._attention_config = self._extract_attention_config()

    def _extract_attention_config(self) -> Dict[str, int]:
        """Extract attention configuration from model metadata."""
        if not self._model_metadata:
            # Return default values if no metadata available
            return {
                'num_query_heads': 32,
                'num_kv_heads': 32,
                'head_size': 128,
                'page_size': 16
            }

        # Extract from architecture section of metadata
        architecture = self._model_metadata.get('architecture', {})

        return {
            'num_query_heads': architecture.get('num_query_heads', 32),
            'num_kv_heads': architecture.get('num_key_value_heads', 32),  # Note: different key name in metadata
            'head_size': architecture.get('head_size', 128),
            'page_size': 16  # This is not in model metadata, it's a runtime parameter
        }

    def initialize(self) -> bool:
        """Initialize Metal backend."""
        start_time = time.perf_counter()

        try:
            # Check if running on macOS
            if sys.platform != 'darwin':
                warnings.warn("Metal backend requires macOS")
                return False

            # Auto-detect metal backend path if not provided
            if self.metal_backend_path is None:
                self.metal_backend_path = self._find_metal_backend_path()

            if self.metal_backend_path is None:
                warnings.warn("Could not find Metal backend path")
                return False

            # Set metallib path
            self._metallib_path = self._find_metallib_path()
            if not self._metallib_path:
                warnings.warn("Could not find Metal library file (.metallib)")
                return False

            # Import and initialize metal_bindings
            try:
                # Add metal backend build path to sys.path temporarily
                build_lib_path = os.path.join(self.metal_backend_path, "build", "lib")
                if build_lib_path not in sys.path:
                    sys.path.insert(0, build_lib_path)

                import metal_bindings
                self._metal_executor = metal_bindings.MetalKernelExecutor(self._metallib_path)

                # Get device info and available kernels
                self._device_info = self._metal_executor.get_device_info()
                available_kernels = self._metal_executor.list_available_kernels()

                # Track which operations we can perform
                self._available_kernels = {
                    'softmax': any('softmax' in kernel.lower() for kernel in available_kernels),
                    'attention': any('attention' in kernel.lower() for kernel in available_kernels),
                    'embedding': any('embedding' in kernel.lower() for kernel in available_kernels),
                    'mlp': any('gemm' in kernel.lower() or 'mlp' in kernel.lower() for kernel in available_kernels),
                    'normalization': any('norm' in kernel.lower() for kernel in available_kernels)
                }

                self.initialization_time = time.perf_counter() - start_time
                self.is_available = True

                print(f"Metal backend initialized successfully")
                print(f"  Device: {self._device_info}")
                print(f"  Available kernels: {len(available_kernels)}")
                print(f"  Metallib: {os.path.basename(self._metallib_path)}")
                return True

            except ImportError as e:
                warnings.warn(f"Failed to import metal_bindings: {e}")
                return False

        except Exception as e:
            warnings.warn(f"Failed to initialize Metal backend: {e}")
            self.increment_error_count()
            return False

    def _find_metal_backend_path(self) -> Optional[str]:
        """Find the Metal backend path automatically."""
        # Try common locations relative to current file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up to pie root and look for backend-metal
        pie_root = current_dir
        for _ in range(10):  # Safety limit
            pie_root = os.path.dirname(pie_root)
            metal_path = os.path.join(pie_root, "backend", "backend-metal")
            if os.path.exists(metal_path) and os.path.exists(os.path.join(metal_path, "CMakeLists.txt")):
                return metal_path

        # Try environment variable
        if 'PIE_METAL_PATH' in os.environ:
            return os.environ['PIE_METAL_PATH']

        return None

    def _find_metallib_path(self) -> Optional[str]:
        """Find the compiled Metal library file."""
        if not self.metal_backend_path:
            return None

        # Try build/lib/pie_metal_kernels.metallib
        metallib_path = os.path.join(self.metal_backend_path, "build", "lib", "pie_metal_kernels.metallib")
        if os.path.exists(metallib_path):
            return metallib_path

        # Try other possible locations
        possible_paths = [
            os.path.join(self.metal_backend_path, "pie_metal_kernels.metallib"),
            os.path.join(self.metal_backend_path, "build", "pie_metal_kernels.metallib"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def run_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """Run attention computation using Metal kernels."""
        start_time = time.perf_counter()

        try:
            # Validate inputs
            is_valid, error_msg = self.validate_inputs("attention", query, key, value)
            if not is_valid:
                raise ValueError(f"Attention input validation failed: {error_msg}")

            if not self._metal_executor:
                raise RuntimeError("Metal executor not initialized")

            # Check if attention kernels are available
            if not self._available_kernels.get('attention', False):
                raise RuntimeError("Metal attention kernels not available")

            # Use attention configuration from model metadata
            num_query_heads = self._attention_config['num_query_heads']
            num_kv_heads = self._attention_config['num_kv_heads']
            head_size = self._attention_config['head_size']
            page_size = self._attention_config['page_size']

            result = self._metal_executor.execute_attention(
                query.astype(np.float32),
                key.astype(np.float32),
                value.astype(np.float32),
                num_query_heads,
                num_kv_heads,
                head_size,
                page_size
            )

            computation_time = time.perf_counter() - start_time
            self.record_performance("attention", computation_time)

            return TensorComputationResult(
                output=result,
                computation_time=computation_time,
                backend_type=self.backend_type,
                metadata={
                    'operation': 'metal_attention',
                    'input_shape': query.shape,
                    'device': self._device_info,
                    'kernels_available': self._available_kernels.get('attention', False)
                }
            )

        except Exception as e:
            self.increment_error_count()
            raise RuntimeError(f"Metal attention computation failed: {e}")

    def run_attention_with_kv_cache(
        self,
        query: np.ndarray,
        kv_cache: np.ndarray,
        kv_page_indices: Optional[np.ndarray] = None,
        kv_page_indptr: Optional[np.ndarray] = None,
        kv_last_page_lens: Optional[np.ndarray] = None,
        **kwargs
    ) -> TensorComputationResult:
        """
        Run attention computation using L4MA/FlashInfer KV cache layout.

        This method calls the Metal kernel with the EXACT SAME KV cache layout as L4MA/FlashInfer.

        Args:
            query: Query tensor from FlashInfer [batch*seq, num_heads, head_size]
            kv_cache: KV cache tensor from L4MA in paged format
            kv_page_indices: Page indices for KV cache
            kv_page_indptr: Page pointers for KV cache
            kv_last_page_lens: Last page lengths for KV cache
            **kwargs: Additional parameters

        Returns:
            TensorComputationResult with attention output using REAL FlashInfer inputs
        """
        start_time = time.perf_counter()

        try:
            if not self._metal_executor:
                raise RuntimeError("Metal executor not initialized")

            # Check if attention kernels are available
            if not self._available_kernels.get('attention', False):
                raise RuntimeError("Metal attention kernels not available")

            # Convert query to 2D format expected by Metal binding
            if len(query.shape) == 3:  # [batch*seq, num_heads, head_size]
                batch_seq, num_heads, head_size = query.shape
                query_2d = query.reshape(batch_seq, num_heads * head_size)
            elif len(query.shape) == 2:  # Already [batch*seq, num_heads * head_size]
                query_2d = query
                batch_seq = query.shape[0]
            else:
                raise ValueError(f"Unsupported query shape: {query.shape}")

            if kv_page_indices is None or kv_page_indptr is None or kv_last_page_lens is None:
                kv_page_indices = np.array([0], dtype=np.int32)
                kv_page_indptr = np.array([0, 1], dtype=np.int32)
                kv_last_page_lens = np.array([batch_seq], dtype=np.int32)

            query_f32 = np.asarray(query_2d, dtype=np.float32, order="C")
            kv_f32 = np.asarray(kv_cache, dtype=np.float32, order="C")

            result = self._metal_executor.execute_attention_with_kv_cache(
                query_f32,
                kv_f32,
                kv_page_indices.astype(np.int32),
                kv_page_indptr.astype(np.int32),
                kv_last_page_lens.astype(np.int32),
                self._attention_config['num_query_heads'],
                self._attention_config['num_kv_heads'],
                self._attention_config['head_size'],
                self._attention_config['page_size']
            )

            computation_time = time.perf_counter() - start_time
            self.record_performance("attention_kv_cache", computation_time)

            print(f"✅ Metal KV cache attention completed:")
            print(f"   Query: {query.shape} → Output: {result.shape}")
            print(f"   KV cache: {kv_cache.shape}")
            print(f"   Computation time: {computation_time:.4f}s")

            return TensorComputationResult(
                output=result,
                computation_time=computation_time,
                backend_type=self.backend_type,
                metadata={
                    'operation': 'metal_attention_kv_cache',
                    'query_shape': query.shape,
                    'kv_cache_shape': kv_cache.shape,
                    'device': self._device_info,
                    'kernels_available': self._available_kernels.get('attention', False),
                    'use_real_kv_cache': True
                }
            )

        except Exception as e:
            self.increment_error_count()
            raise RuntimeError(f"Metal KV cache attention computation failed: {e}")

    def run_mlp(
        self,
        hidden_states: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """Run MLP computation using Metal kernels."""
        start_time = time.perf_counter()

        try:
            # Validate inputs
            is_valid, error_msg = self.validate_inputs("mlp", hidden_states)
            if not is_valid:
                raise ValueError(f"MLP input validation failed: {error_msg}")

            if not self._metal_executor:
                raise RuntimeError("Metal executor not initialized")

            # Check if MLP kernels are available
            if not self._available_kernels.get('mlp', False):
                raise RuntimeError("Metal MLP kernels not available")

            # Execute Metal MLP kernel (simplified for debug framework)
            # The C++ bindings handle the MLP computation internally
            result = self._metal_executor.execute_mlp(
                hidden_states.astype(np.float32)
            )

            computation_time = time.perf_counter() - start_time
            self.record_performance("mlp", computation_time)

            return TensorComputationResult(
                output=result,
                computation_time=computation_time,
                backend_type=self.backend_type,
                metadata={
                    'operation': 'metal_mlp',
                    'input_shape': hidden_states.shape,
                    'device': self._device_info,
                    'kernels_available': self._available_kernels.get('mlp', False)
                }
            )

        except Exception as e:
            self.increment_error_count()
            raise RuntimeError(f"Metal MLP computation failed: {e}")

    def run_embedding(
        self,
        input_ids: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """Run embedding lookup using Metal kernels."""
        start_time = time.perf_counter()

        try:
            # Validate inputs
            is_valid, error_msg = self.validate_inputs("embedding", input_ids)
            if not is_valid:
                raise ValueError(f"Embedding input validation failed: {error_msg}")

            if not self._metal_executor:
                raise RuntimeError("Metal executor not initialized")

            # Check if embedding kernels are available
            if not self._available_kernels.get('embedding', False):
                raise RuntimeError("Metal embedding kernels not available")

            # Get embedding parameters
            hidden_size = kwargs.get('hidden_size', 4096)
            vocab_size = kwargs.get('vocab_size', 32768)
            embedding_table = kwargs.get('embedding_table')

            if embedding_table is None:
                raise ValueError("Embedding table is required for Metal embedding operation")

            # Execute Metal embedding kernel
            result = self._metal_executor.execute_embedding(
                input_ids.astype(np.int32),
                embedding_table.astype(np.float32)
            )

            computation_time = time.perf_counter() - start_time
            self.record_performance("embedding", computation_time)

            return TensorComputationResult(
                output=result,
                computation_time=computation_time,
                backend_type=self.backend_type,
                metadata={
                    'operation': 'metal_embedding',
                    'input_shape': input_ids.shape,
                    'output_shape': result.shape,
                    'device': self._device_info,
                    'kernels_available': self._available_kernels.get('embedding', False)
                }
            )

        except Exception as e:
            self.increment_error_count()
            raise RuntimeError(f"Metal embedding computation failed: {e}")

    def run_normalization(
        self,
        hidden_states: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """Run normalization computation using Metal kernels."""
        start_time = time.perf_counter()

        try:
            if not self._metal_executor:
                raise RuntimeError("Metal executor not initialized")

            # Check if normalization kernels are available
            if not self._available_kernels.get('normalization', False):
                raise RuntimeError("Metal normalization kernels not available")

            # Execute Metal RMS normalization kernel
            eps = kwargs.get('eps', 1e-6)
            result = self._metal_executor.execute_rms_norm(
                hidden_states.astype(np.float32),
                eps
            )

            computation_time = time.perf_counter() - start_time
            self.record_performance("normalization", computation_time)

            return TensorComputationResult(
                output=result.astype(np.float32),
                computation_time=computation_time,
                backend_type=self.backend_type,
                metadata={
                    'operation': 'metal_normalization',
                    'input_shape': hidden_states.shape,
                    'eps': eps,
                    'device': self._device_info,
                    'kernels_available': self._available_kernels.get('normalization', False)
                }
            )

        except Exception as e:
            self.increment_error_count()
            raise RuntimeError(f"Metal normalization computation failed: {e}")

    def execute_softmax(self, input_array: np.ndarray) -> np.ndarray:
        """
        Execute softmax operation using Metal kernels.

        This is an additional method specific to the Metal backend
        that demonstrates direct kernel execution.
        """
        if not self._metal_executor:
            raise RuntimeError("Metal executor not initialized")

        if not self._available_kernels.get('softmax', False):
            raise RuntimeError("Metal softmax kernels not available")

        try:
            return self._metal_executor.execute_softmax(input_array.astype(np.float32))
        except Exception as e:
            raise RuntimeError(f"Metal softmax execution failed: {e}")

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Metal backend capabilities and status."""
        base_capabilities = super().get_capabilities()

        metal_specific = {
            'device_info': self._device_info,
            'metallib_path': self._metallib_path,
            'available_kernels': self._available_kernels.copy(),
            'metal_backend_path': self.metal_backend_path,
            'platform_support': sys.platform == 'darwin'
        }

        base_capabilities.update(metal_specific)
        return base_capabilities

    def cleanup(self):
        """Cleanup Metal backend resources."""
        if self._metal_executor:
            # MetalKernelExecutor handles its own cleanup in destructor
            self._metal_executor = None

        # Clear cached data
        self._available_kernels.clear()
        self._device_info = None
        self._metallib_path = None

        print("Metal backend cleaned up successfully")

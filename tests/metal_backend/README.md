# Integration tests

## Backend tests
All test run inside the root pie directory, not this test directory.

Comprehensive token generation test:
```
tests/metal_backend/test_metal_server_direct.py::test_metal_server_direct_communication -v -s
```

Direct token generation through PIE launched backend:
```
uv run pytest tests/metal_backend/test_metal_server_direct.py::test_metal_server_direct_communication -v -s
```

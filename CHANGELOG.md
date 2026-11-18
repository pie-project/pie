# Changelog

## Unreleased

### Removed

- `engine.activation_dtype`: the engine computes in the checkpoint's dtype.
  A config that still states it is refused at load, naming its file, line
  and key; delete the line:

  ```text
  ~/.pie/config.toml:7: unknown key `engine.activation_dtype`; delete it (it isn't a setting)
  ```

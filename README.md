<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://pie-project.org/img/pie-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://pie-project.org/img/pie-light.svg">
    <img alt="Pie: Programmable serving system for emerging LLM applications"
         src="https://pie-project.org/img/pie-light.svg"
         width="30%">
    <p></p>
  </picture>
</div>

**Pie** is a programmable LLM serving system for custom inference logic,
stateful agents, and serving-side optimization.

> **Note**
>
> Pie is pre-release software under active development. It is best suited
> for testing and research right now.

Documentation, installation instructions, and examples live at
[pie-project.org](https://pie-project.org/).

## Quick demo

```bash
curl -fsSL https://pie-project.org/install.sh | bash
pie config init
pie run text-completion --prompt "The capital of France is"
```

Questions and bug reports are welcome on
[GitHub Issues](https://github.com/pie-project/pie/issues) and
[GitHub Discussions](https://github.com/pie-project/pie/discussions).

## License

[Apache License 2.0](LICENSE)

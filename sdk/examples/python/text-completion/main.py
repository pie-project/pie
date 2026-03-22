"""
Text completion example using inferlet.

This example demonstrates:
- Loading models with get_auto_model()
- Using Context for conversation management
- Text generation with set_return()

Note: send() for streaming is currently not working with Python inferlets.
Use set_return() for non-streaming output instead.
"""

from inferlet import Context, get_auto_model, get_arguments, set_return


def main() -> None:
    # Parse arguments
    args = get_arguments()
    prompt = args.get("prompt", "What is the capital of France?")
    max_tokens = int(args.get("max_tokens", 256))
    system_prompt = args.get("system", "You are a helpful assistant.")

    # Load model
    model = get_auto_model()

    # Generate
    with Context(model) as ctx:
        ctx.system(system_prompt)
        ctx.user(prompt)

        result = ctx.generate(max_tokens=max_tokens, stream=False)
        set_return(result.text)


if __name__ == "__main__":
    main()

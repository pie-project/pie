from inferlet.eta import ForwardPass, Pipeline, StateGeometry, intrinsics, reduce_argmax

async def main(input: dict) -> dict:
    ws = input["ws"]
    pipe = Pipeline()
    tokens, drawn = input["tokens"], []
    for _ in range(input["max_tokens"]):
        fwd = ForwardPass()
        fwd.embed(tokens)
        fwd.state([ws], StateGeometry.folding())
        out = fwd.epilogue(lambda: reduce_argmax(intrinsics.output(0, "logits")))
        pipe.submit(fwd)
        tokens = [await out]
        drawn.append(tokens[0])
    pipe.close()
    ws.give("ws")
    return {"tokens": drawn}

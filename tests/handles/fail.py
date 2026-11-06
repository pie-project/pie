from inferlet.eta import ForwardPass, Pipeline, StateGeometry, intrinsics, reduce_argmax


async def main(input: dict) -> dict:
    ws, pipe = input["ws"], Pipeline()
    failed = None
    try:
        fwd = ForwardPass()
        fwd.embed(input["tokens"])
        fwd.state([ws], StateGeometry.folding())
        out = fwd.epilogue(lambda: reduce_argmax(intrinsics.output(0, "logits")))
        pipe.submit(fwd)
        await out
    except Exception as e:
        failed = str(e)
    ws.give("ws")
    return {"failed": failed}

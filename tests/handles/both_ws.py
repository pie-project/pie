async def main(input: dict) -> dict:
    input["a"].give("a")
    input["b"].give("b")
    return {"ok": True}

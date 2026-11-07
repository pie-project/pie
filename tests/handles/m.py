from inferlet.grammar import Matcher

async def main(input: dict) -> dict:
    m = input["m"]
    for t in input["accept"]:
        try:
            m.accept_tokens([t])
        except Exception:
            break
    term = m.is_terminated()
    n = len(m.mask())
    m.give("m")
    return {"terminated": term, "mask": n}

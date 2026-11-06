from inferlet import frames, media


async def main(input: dict) -> dict:
    f = input["image"]
    span = media.Image.from_frames(f)
    out = {"width": f.width, "height": f.height, "tokens": span.token_count()}
    f.give("image")
    return out

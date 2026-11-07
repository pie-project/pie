from inferlet import frames


async def main(input: dict) -> dict:
    w, h = input["w"], input["h"]
    rgb = bytes((x * 7 + y * 3 + c * 50) % 251 for y in range(h) for x in range(w) for c in range(3))
    frames.Frames.from_rgb8(rgb, w, h).give("out")
    frames.Pcm.from_f32([0.0, 0.25, -0.25, 0.5] * 100, 16000, 1).give("sound")
    return {}

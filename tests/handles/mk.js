import { frames } from '@pie-project/inferlet';

export function main(input) {
  const { w, h } = input;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0; i < rgb.length; i++) rgb[i] = i % 251;
  frames.Frames.fromRgb8(rgb, w, h).give('out');
  frames.Pcm.fromF32(new Float32Array(400), 16000).give('sound');
  return {};
}

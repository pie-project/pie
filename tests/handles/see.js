import { media } from '@pie-project/inferlet';

export function main(input) {
  const f = input.image;
  const span = media.Image.fromFrames(f);
  const out = { width: f.width, height: f.height, tokens: span.tokenCount() };
  f.give('image');
  return out;
}

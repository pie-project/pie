fn pie_tanh(x: f32) -> f32 {
    return tanh(clamp(x, -16.0, 16.0));
}

fn pie_log1p(t: f32) -> f32 {
    if (abs(t) < 0.0625) {
        return t * (1.0 - t * (0.5 - t * (0.33333333333 - t * (0.25 - t * 0.2))));
    }
    return log(1.0 + t);
}

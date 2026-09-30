//! The LSTM step, scalar.
//!
//! Exists because Tacotron2 is the first model here with a recurrence: a
//! bidirectional encoder over the text and two `LSTMCell`s stepped once per mel
//! frame. Everything else in this crate is feed-forward.
//!
//! What it refuses to do: the matmuls. `weight_ih @ x` and `weight_hh @ h` are
//! [`crate::linear()`] and are computed by the caller, because the input side of
//! a whole sequence is one projection while the hidden side is one per step,
//! and fusing them here would force the encoder to do the slow thing.

/// One LSTM step, from the two pre-activation halves.
///
/// `gi` is `weight_ih @ x + bias_ih` and `gh` is `weight_hh @ h + bias_hh`,
/// each `[4 * hidden]` laid out in PyTorch's gate order: input, forget, cell,
/// output. `c` is updated in place, `h` is written, and both are `[hidden]`.
///
/// PyTorch carries two biases rather than one because cuDNN's interface does.
/// They are summed and it makes no difference which side holds what, so this
/// takes the two sums the caller already has rather than asking for four.
pub fn lstm_gates(gi: &[f32], gh: &[f32], c: &mut [f32], h: &mut [f32], hidden: usize) {
    assert_eq!(gi.len(), 4 * hidden, "gi is not [4 * hidden]");
    assert_eq!(gh.len(), 4 * hidden, "gh is not [4 * hidden]");
    assert_eq!(c.len(), hidden, "c is not [hidden]");
    assert_eq!(h.len(), hidden, "h is not [hidden]");

    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    for j in 0..hidden {
        let i_gate = sigmoid(gi[j] + gh[j]);
        let f_gate = sigmoid(gi[hidden + j] + gh[hidden + j]);
        let g_gate = (gi[2 * hidden + j] + gh[2 * hidden + j]).tanh();
        let o_gate = sigmoid(gi[3 * hidden + j] + gh[3 * hidden + j]);
        let cell = f_gate * c[j] + i_gate * g_gate;
        c[j] = cell;
        h[j] = o_gate * cell.tanh();
    }
}

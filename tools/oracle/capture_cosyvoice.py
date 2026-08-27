#!/usr/bin/env python3
"""Capture CosyVoice3's stage boundaries, plus the speaker tensors.

    PYTHONPATH=~/CosyVoice:~/CosyVoice/third_party/Matcha-TTS \
    CUDA_VISIBLE_DEVICES=2 ~/miniconda3/envs/cosyvoice/bin/python \
        tools/oracle/capture_cosyvoice.py \
        --model-dir models/tts/cosyvoice3-0.5b \
        --speaker <ref16k.wav> --out .golden/cosyvoice

Must run in the `cosyvoice` conda env: its torch and transformers pins are not
the ones the rest of this workspace's tooling uses.

# Why the speaker tensors are a *capture* and not a stage

`inference_instruct2` needs three things derived from the reference clip, and
two of them come from onnxruntime models - `campplus.onnx` for the 192-wide
speaker embedding and `speech_tokenizer_v3.onnx` for the prompt speech tokens.
Porting an ONNX graph is a different kind of work from porting a checkpoint:
there is no reference *source* to read, only a graph to reverse. It is also
work that buys nothing here, because the reference speaker never changes. One
clip, pinned at startup, for every utterance the service will ever speak.

So they are extracted once, here, and written as safetensors. The engine loads
them the way it loads any other tensor and never learns that onnxruntime
exists. Re-run this script to change voices; that is the whole cost.

`prompt_speech_feat` is captured too even though it is an ordinary mel and
could be computed in Rust. Capturing it makes the flow stage testable *before*
a second mel frontend is written and verified, which is the wrong order to be
forced into - and it is 200 kB.

# What `frontend_instruct2` does that is easy to get wrong

It builds the zero-shot inputs and then **deletes the LLM's prompt speech
tokens**. So in instruct mode the LLM sees the instruct text and the target
text and no audio prompt at all, while the flow sees the audio prompt and not
the instruct. Wiring the prompt tokens into the LLM because the zero-shot path
has them there gives a model that runs, sounds wrong, and matches on no stage.

# The taps

Stage boundaries first, because they are what let each stage be built and
verified on its own:

    text -> [tokenizer] -> text ids
         -> [llm]       -> speech tokens (6561-way, 25 Hz)
         -> [flow]      -> mel (80 x N, 50 Hz)
         -> [hift]      -> waveform (24 kHz)

Every one is written as its own `.npy` beside a manifest, so a Rust stage can
be diffed against its own input and output rather than against the end of the
pipeline. "The audio is wrong" is not a fact anyone can act on; "the mel
diverges at frame 12" is.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch


def save(out: pathlib.Path, name: str, t) -> dict:
    """Writes one tensor and returns what the manifest should say about it."""
    a = t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.asarray(t)
    # C order, always. A tap that is a transposed *view* - `h.transpose(1, 2)`,
    # say - keeps its original memory, and `np.save` faithfully records that as
    # `fortran_order: True`. The shape in the header then says one thing and the
    # bytes say another, and a reader that trusts the shape gets a permutation
    # of the right values: identical rms, correlation near nothing. Cost is one
    # copy of a tensor that is already on the host.
    a = np.ascontiguousarray(a)
    np.save(out / f"{name}.npy", a)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        # Recorded so a mismatch can be localised before anything is loaded:
        # two tensors with the same shape and a wildly different mean are a
        # different tap, not a numerical difference.
        "mean": float(a.mean()) if a.size else 0.0,
        "absmax": float(np.abs(a).max()) if a.size else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--speaker", required=True, help="the reference wav, 16 kHz")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--text",
        default="台北今仔日好天，溫度差不多二十五度。",
        help="Han text; this backend does not read POJ",
    )
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from cosyvoice.cli.cosyvoice import AutoModel

    # The instruct string the daemon pins. Captured with it rather than
    # without, because it is part of the prompt the LLM sees and a capture
    # taken under a different instruction is a capture of a different model
    # input.
    INSTRUCT = "You are a helpful assistant. 請用閩南話表達。<|endofprompt|>"
    SPK = "taigi-ref"

    torch.manual_seed(1986)
    m = AutoModel(model_dir=a.model_dir, fp16=False)
    m.add_zero_shot_spk(INSTRUCT, a.speaker, SPK)

    fe = m.frontend
    inp = fe.frontend_instruct2(a.text, INSTRUCT, a.speaker, m.sample_rate, SPK)

    manifest = {
        "text": a.text,
        "instruct": INSTRUCT,
        "speaker_wav": os.path.abspath(a.speaker),
        "sample_rate": int(m.sample_rate),
        "tensors": {},
    }
    T = manifest["tensors"]

    # 1. The frontend's outputs, which are the engine's inputs.
    for key in [
        "text",
        "prompt_text",
        "flow_prompt_speech_token",
        "prompt_speech_feat",
        "flow_embedding",
        "llm_embedding",
    ]:
        if key in inp and torch.is_tensor(inp[key]):
            T[key] = save(out, key, inp[key])
            print(f"  {key:26} {T[key]['shape']}")

    # `frontend_instruct2` deletes these; asserting it rather than trusting the
    # reading of the source, since the whole LLM prompt layout depends on it.
    assert "llm_prompt_speech_token" not in inp, "instruct2 should drop the LLM audio prompt"

    dev = m.model.device

    # 2. The LLM, on its own. `llm.inference` is a generator of one token at a
    #    time, which is also what makes it samplable - so the seed is pinned
    #    right before it and the sampler's own knobs come from the yaml.
    #
    # `prompt_text` is the instruct string's ids, and it is **not optional**:
    # CosyVoice3LM asserts that `<|endofprompt|>` (151646) appears in the
    # concatenation of `prompt_text` and `text`, and the instruct is where it
    # comes from. The prompt the model actually sees is
    #
    #     [sos] + embed(prompt_text + text) + [task_id]
    #
    # with the speech-token slot empty, because instruct2 dropped it.
    llm = m.model.llm
    manifest["llm"] = {
        "sos": int(llm.sos),
        "task_id": int(llm.task_id),
        "speech_token_size": int(llm.speech_token_size),
        "endofprompt": 151646,
    }
    print(f"  llm markers: {manifest['llm']}")

    torch.manual_seed(1986)
    with torch.no_grad():
        tokens = list(
            llm.inference(
                text=inp["text"].to(dev),
                text_len=torch.tensor([inp["text"].shape[1]], dtype=torch.int32).to(dev),
                prompt_text=inp["prompt_text"].to(dev),
                prompt_text_len=torch.tensor(
                    [inp["prompt_text"].shape[1]], dtype=torch.int32
                ).to(dev),
                prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32).to(dev),
                prompt_speech_token_len=torch.tensor([0], dtype=torch.int32).to(dev),
                embedding=inp["llm_embedding"].to(dev),
            )
        )
    speech_token = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0)
    T["speech_token"] = save(out, "speech_token", speech_token)
    print(f"  {'speech_token':26} {T['speech_token']['shape']}  {tokens[:12]}...")

    # 2b. The same LLM, **teacher-forced**, which is the tap that can actually
    #     be tested against.
    #
    # `ras_sampling` draws with `torch.multinomial`, so the token sequence above
    # is a function of PyTorch's RNG as much as of the weights. Reproducing that
    # RNG bit-for-bit in Rust is not a reasonable thing to attempt, and a
    # comparison of two different draws from the same distribution proves
    # nothing either way.
    #
    # So the captured tokens are fed back in and the log-probabilities recorded
    # at every step. That is deterministic, it is 143 observations instead of
    # one, and it keeps measuring past the first divergence - the same reason
    # the chat model is compared this way. The sampler is then a separate
    # question, tested against the distribution rather than against a draw.
    with torch.no_grad():
        sos_emb = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
        task_emb = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
        text_all = torch.concat([inp["prompt_text"], inp["text"]], dim=1).to(dev)
        text_emb = llm.llm.model.model.embed_tokens(text_all)
        # The forced continuation: every speech token but the last, since the
        # last one's own prediction is the step after it and has no target.
        forced = llm.speech_embedding(speech_token[:, :-1].long().to(dev))
        lm_input = torch.concat([sos_emb, text_emb, task_emb, forced], dim=1)

        n = lm_input.shape[1]
        y, _ = llm.llm.forward_one_step(
            lm_input,
            masks=torch.tril(torch.ones((1, n, n), device=dev)).to(torch.bool),
            cache=None,
        )
        # One row per speech token: the position that predicts token `i` is the
        # last prompt position for i = 0, then one per forced token after it.
        head = 1 + text_emb.shape[1] + 1
        logp = llm.llm_decoder(y[:, head - 1 :]).log_softmax(dim=-1)

    T["llm_prompt_len"] = {"shape": [], "dtype": "int", "mean": head, "absmax": head}
    T["forced_logprobs"] = save(out, "forced_logprobs", logp)
    argmax = logp.argmax(dim=-1).squeeze(0).cpu()
    agree = int((argmax == speech_token.squeeze(0).long()).sum())
    print(
        f"  {'forced_logprobs':26} {T['forced_logprobs']['shape']}  "
        f"argmax agrees with the sampled run at {agree}/{speech_token.shape[1]} "
        f"(the rest is what sampling is for)"
    )

    # 3. The flow, from those tokens. Deterministic given its input: the CFM
    #    solver draws one noise tensor, so that is captured too and fed to the
    #    Rust side rather than re-drawn - a solver compared against a different
    #    starting point is not a comparison.
    with torch.no_grad():
        mel, _ = m.model.flow.inference(
            token=speech_token.to(dev),
            token_len=torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(dev),
            prompt_token=inp["flow_prompt_speech_token"].to(dev),
            prompt_token_len=torch.tensor(
                [inp["flow_prompt_speech_token"].shape[1]], dtype=torch.int32
            ).to(dev),
            prompt_feat=inp["prompt_speech_feat"].to(dev),
            prompt_feat_len=torch.tensor(
                [inp["prompt_speech_feat"].shape[1]], dtype=torch.int32
            ).to(dev),
            embedding=inp["flow_embedding"].to(dev),
            streaming=False,
            finalize=True,
        )
    T["mel"] = save(out, "mel", mel)
    print(f"  {'mel':26} {T['mel']['shape']}")

    # 3b. Inside the flow, boundary by boundary, plus the noise the solver
    #     starts from.
    #
    # `CausalConditionalCFM.__init__` seeds torch and draws
    # `randn([1, 80, 15000])`, so the starting point is a construction-time
    # buffer like the vocoder's dither - deterministic given that seed, and not
    # something to reproduce in Rust. A solver compared against a different
    # starting point is not a comparison, so it is captured.
    flow = m.model.flow
    with torch.no_grad():
        emb = torch.nn.functional.normalize(inp["flow_embedding"].to(dev), dim=1)
        emb = flow.spk_embed_affine_layer(emb)
        T["spk80"] = save(out, "spk80", emb)

        tok = torch.concat(
            [inp["flow_prompt_speech_token"].to(dev), speech_token.to(dev)], dim=1
        )
        te = flow.input_embedding(torch.clamp(tok, min=0))
        T["flow_token_embed"] = save(out, "flow_token_embed", te)

        h = flow.pre_lookahead_layer(te)
        T["pre_lookahead"] = save(out, "pre_lookahead", h)
        h = h.repeat_interleave(flow.token_mel_ratio, dim=1)
        # Channel-major, which is how `solve_euler` hands it to the estimator
        # (`mu_in[0] = mu` where `mu = h.transpose(1, 2)`). Saving `h` as it
        # stands here would be position-major and would silently transpose the
        # estimator's largest input.
        T["flow_mu"] = save(out, "flow_mu", h.transpose(1, 2))

        mel_len1 = inp["prompt_speech_feat"].shape[1]
        n = h.shape[1]
        conds = torch.zeros([1, n, 80], device=dev, dtype=h.dtype)
        conds[:, :mel_len1] = inp["prompt_speech_feat"].to(dev)
        T["flow_cond"] = save(out, "flow_cond", conds.transpose(1, 2))

        T["cfm_noise"] = save(out, "cfm_noise", flow.decoder.rand_noise[:, :, :n])
        print(f"  {'cfm_noise':26} {T['cfm_noise']['shape']}  (constructed, not in flow.pt)")

        # One estimator evaluation at the solver's first timestep, with the
        # classifier-free pair batched exactly as `solve_euler` batches it.
        est = flow.decoder.estimator
        t_span = 1 - torch.cos(torch.linspace(0, 1, 11, device=dev) * 0.5 * torch.pi)
        z = flow.decoder.rand_noise[:, :, :n].to(dev)
        mask = torch.ones(1, 1, n, device=dev)
        x_in = torch.zeros(2, 80, n, device=dev)
        mu_in = torch.zeros(2, 80, n, device=dev)
        t_in = torch.zeros(2, device=dev)
        spk_in = torch.zeros(2, 80, device=dev)
        cond_in = torch.zeros(2, 80, n, device=dev)
        x_in[:] = z
        mu_in[0] = h.transpose(1, 2)
        t_in[:] = t_span[0]
        spk_in[0] = emb
        cond_in[0] = conds.transpose(1, 2)
        mask_in = torch.ones(2, 1, n, device=dev)

        T["t_span"] = save(out, "t_span", t_span)

        # Inside the estimator, so a failure names a module rather than "the
        # mel is wrong". Row 0 only: the two rows differ in their inputs, not
        # in their arithmetic.
        xt = x_in.transpose(1, 2)
        mut = mu_in.transpose(1, 2)
        ct = cond_in.transpose(1, 2)
        temb = est.time_embed(t_in)
        T["dit_temb"] = save(out, "dit_temb", temb)
        from einops import repeat as _repeat
        _cat = torch.cat(
            [xt, ct, mut, _repeat(spk_in, "b c -> b t c", t=xt.shape[1])], dim=-1
        )
        T["dit_cat"] = save(out, "dit_cat", _cat)
        _proj = est.input_embed.proj(_cat)
        T["dit_proj"] = save(out, "dit_proj", _proj)
        T["dit_pos"] = save(out, "dit_pos", est.input_embed.conv_pos_embed(_proj))
        ie = est.input_embed(xt, ct, mut, spk_in)
        T["dit_input_embed"] = save(out, "dit_input_embed", ie)

        rope = est.rotary_embed.forward_from_seq_len(n)
        am = torch.ones(2, 1, n, n, dtype=torch.bool, device=dev)
        hblk = ie
        for bi, blk in enumerate(est.transformer_blocks):
            hblk = blk(hblk, temb, mask=am, rope=rope)
            if bi in (0, 1, 7, 14, 21):
                T[f"dit_block{bi}"] = save(out, f"dit_block{bi}", hblk)
        print(f"  {'dit_input_embed':26} {T['dit_input_embed']['shape']}")

        dphi = est(x_in, mask_in, mu_in, t_in, spk_in, cond_in, streaming=False)
        T["dit_step0"] = save(out, "dit_step0", dphi)
        print(f"  {'dit_step0':26} {T['dit_step0']['shape']}")

    # 4. The vocoder, with its own boundaries.
    #
    # `f0_predictor` is run in **float64** by upstream - "precision is crucial
    # for causal inference" - and then cast back. That is captured as it is
    # rather than corrected: whatever this engine does, it has to reproduce the
    # numbers the reference actually produces.
    hift = m.model.hift
    with torch.no_grad():
        hift.f0_predictor.to(torch.float64)
        f0 = hift.f0_predictor(mel.to(dev).to(torch.float64), finalize=True).to(mel)
        source = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _, _ = hift.m_source(source)
        source = source.transpose(1, 2)
    T["f0"] = save(out, "f0", f0)
    T["source"] = save(out, "source", source)
    print(f"  {'f0':26} {T['f0']['shape']}")
    print(f"  {'source':26} {T['source']['shape']}")

    # The three buffers that are **not in the checkpoint**.
    #
    # `SineGen2` and `SourceModuleHnNSF` each call `torch.rand` in `__init__`
    # and keep the result as a plain attribute - not a parameter, not a
    # registered buffer - so it never reaches `hift.pt`. They are regenerated
    # from torch's global RNG every time the model is constructed, which means
    # upstream itself does not reproduce across load orderings.
    #
    # They are dither: an initial phase offset per harmonic, a bank of sine
    # noise, and the unvoiced noise. Perceptually they are nothing; numerically
    # they decide every sample. So a comparison against the reference has to be
    # fed the same ones, and they are captured here, sliced to this utterance
    # rather than at their full 300 seconds - 288 MB against 5 MB.
    gen = hift.m_source.l_sin_gen
    n = source.shape[2]
    for name, t in [
        ("sine_rand_ini", gen.rand_ini),
        ("sine_waves", gen.sine_waves[:, :n]),
        ("uv_noise", hift.m_source.uv[:, :n]),
    ]:
        T[name] = save(out, name, t)
        print(f"  {name:26} {T[name]['shape']}  (constructed, not in hift.pt)")

    # Inside `decode`, stage by stage. "The waveform is wrong" localises to
    # nothing; "stage 1 is wrong" localises to twelve tensors.
    with torch.no_grad():
        import torch.nn.functional as F

        sr_, si_ = hift._stft(source.to(dev).squeeze(1))
        s_stft = torch.cat([sr_, si_], dim=1)
        T["s_stft"] = save(out, "s_stft", s_stft)
        print(f"  {'s_stft':26} {T['s_stft']['shape']}")

        xt = hift.conv_pre(mel.to(dev))
        T["conv_pre"] = save(out, "conv_pre", xt)
        print(f"  {'conv_pre':26} {T['conv_pre']['shape']}")

        for i in range(hift.num_upsamples):
            xt = F.leaky_relu(xt, hift.lrelu_slope)
            xt = hift.ups[i](xt)
            if i == hift.num_upsamples - 1:
                xt = hift.reflection_pad(xt)
            sd = hift.source_downs[i](s_stft)
            T[f"source_down{i}"] = save(out, f"source_down{i}", sd)
            sd = hift.source_resblocks[i](sd)
            xt = xt + sd
            xs = None
            for j in range(hift.num_kernels):
                xs = hift.resblocks[i * hift.num_kernels + j](xt) if xs is None \
                    else xs + hift.resblocks[i * hift.num_kernels + j](xt)
            xt = xs / hift.num_kernels
            T[f"stage{i}"] = save(out, f"stage{i}", xt)
            print(f"  {f'stage{i}':26} {T[f'stage{i}']['shape']}")

        xt = F.leaky_relu(xt)
        xt = hift.conv_post(xt)
        T["conv_post"] = save(out, "conv_post", xt)
        print(f"  {'conv_post':26} {T['conv_post']['shape']}")

    with torch.no_grad():
        wav, _ = hift.inference(speech_feat=mel.to(dev), finalize=True)
    T["wav"] = save(out, "wav", wav)
    print(f"  {'wav':26} {T['wav']['shape']}  {wav.shape[1] / m.sample_rate:.2f}s")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}/manifest.json: {len(T)} tensors")


if __name__ == "__main__":
    main()

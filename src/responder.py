"""Short in-character replies from a small local LLM (Qwen3-1.7B, 4-bit, MLX).

MLX is Apple's array framework. mlx-lm runs LLMs on the M-series GPU, and
it's the fastest local option on this hardware. 4-bit quantization stores
each weight in 4 bits instead of 16, so the model needs ~1 GB of RAM instead
of ~3.4 GB. The *parameter count* is still ~2.0B.

Grounding: the prompt hands the LLM the transcript, the fused emotion with
its confidence, and (when they disagree) what the words vs. the tone
suggested. Low confidence -> we explicitly tell it NOT to assume an emotion.
"""
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from src.config import LLM_MLX

PERSONA = (
    "You are Lumi, a small expressive desk-lamp robot. You are warm, playful and "
    "curious. You reply out loud in ONE or TWO short sentences (max 30 words), "
    "plain spoken English, no emojis, no lists, no stage directions."
)

LOW_CONFIDENCE = 0.45   # below this, don't trust the emotion label


def build_user_message(state: dict, history: list[tuple[str, str]]) -> str:
    lines = []
    if history:
        lines.append("Recent conversation:")
        for user, bot in history[-3:]:
            lines.append(f"  Person: {user}")
            lines.append(f"  Lumi: {bot}")
    lines.append(f'The person just said: "{state["transcript"]}"')
    emo, conf = state["emotion"], state["confidence"]
    if conf < LOW_CONFIDENCE:
        lines.append(f"Emotion detector is UNSURE (best guess {emo}, confidence {conf:.2f}). "
                     "Do not assume how they feel; respond in a neutral, gently curious way.")
    else:
        lines.append(f"Emotion detector: they sound {emo} (confidence {conf:.2f}).")
        pm = state.get("per_modality", {})
        if "text" in pm and "audio" in pm:
            t, a = max(pm["text"], key=pm["text"].get), max(pm["audio"], key=pm["audio"].get)
            if t != a and pm["audio"][a] >= LOW_CONFIDENCE:
                lines.append(f"Note: their words alone read as {t}, but their tone of voice sounds {a}. "
                             "Trust the tone.")
        if "face" in pm:
            f = max(pm["face"], key=pm["face"].get)
            if f != emo and pm["face"][f] >= 0.5:
                lines.append(f"Their facial expression looks {f}, which differs from what they said. "
                             "You may gently acknowledge that.")
    lines.append("Reply to them, reacting to both what they said and how they feel.")
    return "\n".join(lines)


def strip_emoji(s: str) -> str:
    """The persona says 'no emojis' but a 1.7B model sometimes adds one anyway.
    A robot speaks its reply, so drop pictographs (code points >= U+2600)."""
    return "".join(c for c in s if ord(c) < 0x2600)


class Responder:
    def __init__(self, model_name: str = LLM_MLX, seed: int = 0):
        mx.random.seed(seed)  # sampled replies, but reproducible run to run
        self.model, self.tok = load(model_name)
        self.sampler = make_sampler(temp=0.7, top_p=0.9)

    def stream(self, state: dict, history: list[tuple[str, str]], max_tokens: int = 60):
        """Yields (text_piece, ms_since_start). The first yield = time to first token."""
        messages = [{"role": "system", "content": PERSONA},
                    {"role": "user", "content": build_user_message(state, history)}]
        # enable_thinking=False turns off Qwen3's slow "<think>" reasoning mode.
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
        t0 = time.perf_counter()
        for r in stream_generate(self.model, self.tok, prompt, max_tokens=max_tokens, sampler=self.sampler):
            text = strip_emoji(r.text)
            if text:
                yield text, (time.perf_counter() - t0) * 1000

"""Pure labels and model descriptions."""

import re

from .i18n import TEXTS


def get_full_model_label(res, model_name, lang):
    if res.get("use_first_order_phase1", False):
        if int(res.get("n_phases", 0)) == 1:
            return TEXTS["full_model_first_order_only"][lang]
        return TEXTS["full_model_first_order"][lang].format(model_name)
    return TEXTS["full_model_sigmoidal"][lang].format(model_name)


def format_subscript_label(label):
    """
    Converts underscore notation into subscript-like presentation where possible.
    Example: r_max -> rₘₐₓ
    """
    sub_map = {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        "a": "ₐ",
        "e": "ₑ",
        "h": "ₕ",
        "i": "ᵢ",
        "j": "ⱼ",
        "k": "ₖ",
        "l": "ₗ",
        "m": "ₘ",
        "n": "ₙ",
        "o": "ₒ",
        "p": "ₚ",
        "r": "ᵣ",
        "s": "ₛ",
        "t": "ₜ",
        "u": "ᵤ",
        "v": "ᵥ",
        "x": "ₓ",
    }
    txt = str(label)
    if "_" not in txt:
        return txt
    base, sub = txt.split("_", 1)
    sub_fmt = "".join(sub_map.get(ch, ch) for ch in sub.lower())
    return f"{base}{sub_fmt}"


def normalize_axis_math_label(label_text):
    """
    Accepts user-friendly axis text and converts common superscript inputs
    to Matplotlib mathtext, e.g. lambda^2, lambdaˆ2, λ^2, <sup>2</sup>.
    """
    text = str(label_text or "").strip()
    if not text:
        return text

    # Common copy/paste variants seen in UI input.
    text = text.replace("ˆ", "^")
    text = re.sub(r"(?is)<\s*sup\s*>(.*?)<\s*/\s*sup\s*>", r"^\1", text)

    def _to_unicode_superscript(exp_txt):
        superscript_map = {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
            "+": "⁺",
            "-": "⁻",
        }
        converted = "".join(superscript_map.get(ch, ch) for ch in exp_txt)
        return converted

    def _convert_compact_unit_exponents(raw_text):
        # Converts unit-style tokens such as m2, m-2, cm3, h-1 to Unicode superscript.
        # Restrict base length to avoid changing ordinary words like "phase2".
        unit_pattern = re.compile(r"\b([A-Za-zµμΩωα-ωΑ-Ω]{1,4})([-+]?\d+)\b")

        def _repl(match):
            base = match.group(1)
            exponent = match.group(2)
            return f"{base}{_to_unicode_superscript(exponent)}"

        return unit_pattern.sub(_repl, raw_text)

    text = _convert_compact_unit_exponents(text)

    # If user already provided mathtext, keep it unchanged.
    if "$" in text:
        return text

    has_math_hint = (
        ("^" in text) or ("_" in text) or any(ch in text for ch in ("λ", "μ", "σ", "θ", "φ", "ω"))
    )
    has_greek_word = (
        re.search(
            r"\b(lambda|alpha|beta|gamma|delta|epsilon|mu|sigma|theta|phi|omega)\b",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )
    if not (has_math_hint or has_greek_word):
        return text

    # Replace common Greek names/symbols with mathtext commands.
    greek_symbols = {
        "λ": r"\lambda",
        "μ": r"\mu",
        "σ": r"\sigma",
        "θ": r"\theta",
        "φ": r"\phi",
        "ω": r"\omega",
        "α": r"\alpha",
        "β": r"\beta",
        "γ": r"\gamma",
        "δ": r"\delta",
        "ε": r"\epsilon",
    }
    for symbol, latex_cmd in greek_symbols.items():
        text = text.replace(symbol, latex_cmd)

    greek_words = {
        "lambda": r"\lambda",
        "alpha": r"\alpha",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "delta": r"\delta",
        "epsilon": r"\epsilon",
        "mu": r"\mu",
        "sigma": r"\sigma",
        "theta": r"\theta",
        "phi": r"\phi",
        "omega": r"\omega",
    }
    for word, latex_cmd in greek_words.items():
        text = re.sub(rf"\b{word}\b", lambda _m, rep=latex_cmd: rep, text, flags=re.IGNORECASE)

    # Convert compact exponent after Greek names too (e.g., lambda2 -> lambda^2).
    text = re.sub(
        r"(\\(?:lambda|alpha|beta|gamma|delta|epsilon|mu|sigma|theta|phi|omega))([-+]?\d+)\b",
        r"\1^\2",
        text,
    )

    # Normalize simple exponents like ^2, ^-1, ^10 to brace form.
    text = re.sub(r"\^(-?\d+)", r"^{\1}", text)

    return f"${text}$"

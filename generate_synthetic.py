"""
Generates synthetic dataset of rendered LaTeX lines (Russian math text + formulas).

Usage:
    python generate_synthetic.py
    python generate_synthetic.py --count 10000 --profile rtx4060_8gb
    python generate_synthetic.py --count 200 --force   # quick test run

After generation run:
    python prepare_data.py --datasets synthetic
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import load_config, Config
from data.preprocess import crop_to_content

try:
    from pdf2image import convert_from_path
    _PDF2IMAGE_OK = True
except ImportError:
    _PDF2IMAGE_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Random helpers
# ──────────────────────────────────────────────────────────────────────────────

_VARS   = ["x", "y", "z", "a", "b", "c", "t", "s", "u", "v"]
_FUNCS  = ["f", "g", "h", "F", "G", r"\varphi", r"\psi"]
_GREEKS = [r"\alpha", r"\beta", r"\gamma", r"\delta", r"\lambda",
           r"\mu", r"\varepsilon", r"\xi", r"\eta", r"\theta", r"\omega"]
_SETS   = [r"\mathbb{R}", r"\mathbb{Z}", r"\mathbb{N}", r"\mathbb{Q}", r"\mathbb{C}"]
_NORMS  = [r"\|", r"|"]

def _var():   return random.choice(_VARS)
def _fn():    return random.choice(_FUNCS)
def _gk():    return random.choice(_GREEKS)
def _set():   return random.choice(_SETS)
def _n():     return str(random.randint(1, 15))
def _N():     return str(random.randint(2, 20))
def _pos():   return str(random.randint(1, 9))
def _int():   return str(random.randint(-5, 9))
def _exp():   return random.choice(["2", "3", "n", r"\alpha"])

_GENERATORS: dict[str, callable] = {
    "var": _var, "fn": _fn, "gk": _gk, "set": _set,
    "n": _n, "N": _N, "pos": _pos, "int": _int, "exp": _exp,
}

# Kinds whose value is shared across all occurrences in a single template.
# Mathematical consistency: the same variable/function name must appear identically.
_SHARED_KINDS = ("var", "fn", "gk", "set", "exp")
# Kinds where each occurrence gets an independent random value (coefficients, counts, etc.)
_INDEPENDENT_KINDS = ("n", "N", "pos", "int")

_INDEXED_RE = re.compile(r'<<([a-z]+):(\w+)>>')


def _fill(template: str) -> str:
    """Replace <<kind>> and <<kind:label>> placeholders with random values.

    Rules:
    - <<kind:label>>  Shared by label within the template (same label → same value).
                      Use when two distinct occurrences must have different values
                      but each must be internally consistent, e.g. <<var:A>> / <<var:B>>.
    - <<var>>, <<fn>>, <<gk>>, <<set>>, <<exp>>
                      Shared per template: one value picked, ALL occurrences replaced.
                      Ensures mathematical consistency (integrand and dx use the same var).
    - <<n>>, <<N>>, <<pos>>, <<int>>
                      Independent: each occurrence gets its own random value.
    """
    result = template

    # Pass 1: labelled placeholders <<kind:label>> — shared per (kind, label) pair
    label_cache: dict[str, str] = {}
    def _replace_indexed(m: re.Match) -> str:
        key = m.group(0)
        if key not in label_cache:
            label_cache[key] = _GENERATORS[m.group(1)]()
        return label_cache[key]
    result = _INDEXED_RE.sub(_replace_indexed, result)

    # Pass 2: shared-kind placeholders — one value for the entire template
    for kind in _SHARED_KINDS:
        ph = f"<<{kind}>>"
        if ph in result:
            val = _GENERATORS[kind]()
            result = result.replace(ph, val)

    # Pass 3: independent-kind placeholders — fresh value per occurrence
    for kind in _INDEPENDENT_KINDS:
        ph = f"<<{kind}>>"
        while ph in result:
            result = result.replace(ph, _GENERATORS[kind](), 1)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Template library
# ──────────────────────────────────────────────────────────────────────────────
# Each entry: (content_template, category)
# category: "text" | "formula" | "mixed" | "long"
# Use raw strings — LaTeX braces stay as-is; <<...>> are our placeholders.

TEMPLATES: list[tuple[str, str]] = [

    # ── TEXT-ONLY ─────────────────────────────────────────────────────────────
    (r"Теорема~<<n>>.",                                         "text"),
    (r"Лемма~<<n>>.",                                           "text"),
    (r"Следствие~<<n>>.",                                       "text"),
    (r"Определение~<<n>>.",                                     "text"),
    (r"Доказательство.",                                        "text"),
    (r"Замечание.",                                             "text"),
    (r"Пример~<<n>>.",                                          "text"),
    (r"Доказательство теоремы~<<n>>.",                          "text"),
    (r"Доказательство леммы~<<n>>.",                            "text"),
    (r"Пусть $<<fn>>$ непрерывна на $[a, b]$.",                 "text"),
    (r"Пусть $<<fn>>$ дифференцируема на $<<set>>$.",           "text"),
    (r"Пусть $\varepsilon > 0$ произвольное.",                  "text"),
    (r"Из условия следует, что $<<fn>>$ ограничена сверху.",    "text"),
    (r"Заметим, что правая часть не зависит от $<<var>>$.",     "text"),
    (r"Рассмотрим функцию $<<fn>> : <<set>> \to <<set>>$.",     "text"),
    (r"Не умаляя общности предположим, что $<<var>> > 0$.",     "text"),
    (r"По условию $a_n \to 0$ при $n \to \infty$.",             "text"),
    (r"Из непрерывности $<<fn>>$ следует ограниченность.",      "text"),
    (r"Тогда утверждение теоремы~<<n>> выполнено.",             "text"),
    (r"Пусть задана последовательность $\{a_n\}_{n=1}^{\infty}$.", "text"),

    # ── FORMULA-ONLY ──────────────────────────────────────────────────────────
    (r"$\frac{<<pos>>}{<<pos>>} + \frac{<<pos>>}{<<pos>>}$",    "formula"),
    (r"$\int_0^{<<n>>} <<fn>>(<<var>>)\,d<<var>>$",             "formula"),
    (r"$\int_a^b <<fn>>(<<var>>)\,d<<var>> = F(b) - F(a)$",     "formula"),
    (r"$\int_0^{\infty} e^{-<<var>>^2}\,d<<var>> = \frac{\sqrt{\pi}}{2}$", "formula"),
    (r"$\int_{-\infty}^{+\infty} e^{-<<var>>^2}\,d<<var>> = \sqrt{\pi}$",  "formula"),
    (r"$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$", "formula"),
    (r"$\sum_{n=1}^{<<N>>} \frac{<<pos>>}{n^<<pos>>}$",         "formula"),
    (r"$\sum_{n=0}^{\infty} <<var>>^n = \frac{1}{1-<<var>>},\quad |<<var>>| < 1$", "formula"),
    (r"$\lim_{<<var>> \to 0} \frac{\sin <<var>>}{<<var>>} = 1$","formula"),
    (r"$\lim_{<<var>> \to <<gk>>} <<fn>>(<<var>>) = <<int>>$",  "formula"),
    (r"$\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e$", "formula"),
    (r"$e^{i\pi} + 1 = 0$",                                     "formula"),
    (r"$e^{<<var>>} = \sum_{n=0}^{\infty} \frac{<<var>>^n}{n!}$", "formula"),
    (r"$\sin^2 <<var>> + \cos^2 <<var>> = 1$",                  "formula"),
    (r"$(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^k b^{n-k}$", "formula"),
    (r"$\binom{n}{k} = \frac{n!}{k!(n-k)!}$",                  "formula"),
    (r"$\det(A) \neq 0$",                                       "formula"),
    (r"$\|<<var:A>> + <<var:B>>\| \leq \|<<var:A>>\| + \|<<var:B>>\|$", "formula"),
    (r"$f'(<<var>>) = \lim_{h \to 0} \frac{f(<<var>>+h) - f(<<var>>)}{h}$", "formula"),
    (r"$x_{1,2} = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$",        "formula"),
    (r"$\nabla <<fn>> = \left(\frac{\partial <<fn>>}{\partial x},\, \frac{\partial <<fn>>}{\partial y}\right)$", "formula"),
    (r"$\frac{d}{d<<var>>}\left[\int_a^{<<var>>} <<fn>>(t)\,dt\right] = <<fn>>(<<var>>)$", "formula"),
    (r"$P(A \cup B) = P(A) + P(B) - P(A \cap B)$",             "formula"),
    (r"$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$", "formula"),
    (r"$f(<<var>>) = f(a) + f'(a)(<<var>>-a) + o(<<var>>-a)$", "formula"),
    (r"$a^<<exp>> + b^<<exp>> = c^<<exp>>$",                    "formula"),
    (r"$|<<var>>| = \begin{cases} <<var>>, & <<var>> \geq 0 \\ -<<var>>, & <<var>> < 0 \end{cases}$", "formula"),
    (r"$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$", "formula"),
    (r"$\oint_C \mathbf{F}\,d\mathbf{r} = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right)dA$", "formula"),

    # ── MIXED ─────────────────────────────────────────────────────────────────
    (r"Тогда $<<fn>>'(<<var>>) = <<int>><<var>>$ при $<<var>> > 0$.",  "mixed"),
    (r"Из равенства $a + b = c$ следует $a = c - b$.",                 "mixed"),
    (r"Обозначим $S_n = \sum_{k=1}^n a_k$.",                           "mixed"),
    (r"При $<<var>> = <<n>>$ получаем $<<fn>>(<<var>>) = <<int>>$.",   "mixed"),
    (r"Пусть $A = (a_{ij})$ --- матрица размера $<<n>> \times <<n>>$.", "mixed"),
    (r"По формуле сложения $\sin(x + y) = \sin x \cos y + \cos x \sin y$.", "mixed"),
    (r"Если $\lim_{n \to \infty} a_n = A$, то $\lim_{n \to \infty} ca_n = cA$.", "mixed"),
    (r"Из $|a_n - A| < \varepsilon$ при $n > N$ следует $a_n \to A$.", "mixed"),
    (r"Ряд $\sum_{n=1}^{\infty} \frac{<<pos>>}{n^<<pos>>}$ сходится.", "mixed"),
    (r"При $<<gk>> \to 0$ имеем $\sin <<gk>> \approx <<gk>>$.",        "mixed"),
    (r"Условие Липшица: $|<<fn>>(x) - <<fn>>(y)| \leq L|x - y|$ для всех $x, y$.", "mixed"),
    (r"Докажем, что $\int_0^1 <<var>>^n\,d<<var>> = \frac{1}{n+1}$ для всех $n \geq 0$.", "mixed"),
    (r"По лемме~<<n>>, функция $<<fn>>$ монотонна на $[<<int>>, <<n>>]$.", "mixed"),
    (r"Решение $<<fn>>(<<var>>) = C_1 e^{<<int>><<var>>} + C_2 e^{<<int>><<var>>}$.", "mixed"),
    (r"Пусть $<<fn>>$ имеет производную $<<fn>>'(<<var>>) = <<int>><<var>>^<<n>> - <<pos>>$.", "mixed"),
    (r"Тогда $\|A\| = \sup_{<<var>> \neq 0} \frac{\|A<<var>>\|}{\|<<var>>\|}$.", "mixed"),
    (r"Число $e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n \approx 2{,}718$.", "mixed"),
    (r"Площадь $S = \int_a^b |<<fn>>(<<var>>)|\,d<<var>>$.",           "mixed"),
    (r"Запишем в виде $<<fn>>(<<var>>) = \frac{P(<<var>>)}{Q(<<var>>)}$, где $\deg P < \deg Q$.", "mixed"),
    (r"При $n \to \infty$ последовательность $a_n = \frac{<<n:A>>n + <<pos>>}{<<pos:B>>n - <<pos>>} \to \frac{<<n:A>>}{<<pos:B>>}$.", "mixed"),
    (r"Тогда $\int_0^{\pi} \sin^<<n>> <<var>>\,d<<var>>$ вычисляется рекуррентно.", "mixed"),
    (r"По теореме о среднем: $<<fn>>(b) - <<fn>>(a) = <<fn>>'(c)(b - a)$.", "mixed"),
    (r"Матрица перехода $P$ такова, что $P^{-1}AP = \mathrm{diag}(<<gk>>_1, \ldots, <<gk>>_n)$.", "mixed"),

    # ── LONG (for tail distribution, >200 tokens each) ────────────────────────
    (r"$\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} = \begin{pmatrix} b_1 \\ b_2 \\ b_3 \end{pmatrix}$", "long"),
    (r"Система: $\begin{cases} <<int>>x + <<int>>y + <<int>>z = <<int>> \\ <<int>>x + <<int>>y + <<int>>z = <<int>> \\ <<int>>x + <<int>>y + <<int>>z = <<int>> \end{cases}$", "long"),
    (r"$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n + R_n(x)$", "long"),
    (r"$\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}$", "long"),
    (r"$\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!}$", "long"),
    (r"$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \cdots + \frac{x^n}{n!} + \cdots = \sum_{n=0}^{\infty} \frac{x^n}{n!}$", "long"),
    (r"$\frac{1}{(1-x)^2} = 1 + 2x + 3x^2 + 4x^3 + \cdots + nx^{n-1} + \cdots = \sum_{n=1}^{\infty} nx^{n-1},\quad |x| < 1$", "long"),
    (r"$\det\begin{pmatrix} a_{11} & a_{12} & a_{13} & a_{14} \\ a_{21} & a_{22} & a_{23} & a_{24} \\ a_{31} & a_{32} & a_{33} & a_{34} \\ a_{41} & a_{42} & a_{43} & a_{44} \end{pmatrix}$", "long"),
    (r"Цепочка равенств: $\int_0^1 f(x)\,dx = F(1) - F(0) = \lim_{n\to\infty} \sum_{k=1}^n f\!\left(\frac{k}{n}\right)\frac{1}{n}$.", "long"),
    (r"$\|A + B\|^2 = \langle A+B, A+B\rangle = \|A\|^2 + 2\langle A, B\rangle + \|B\|^2 \leq (\|A\| + \|B\|)^2$", "long"),
    (r"По формуле Ньютона--Лейбница: $<<fn>>(b) - <<fn>>(a) = \int_a^b <<fn>>'(t)\,dt = \int_a^b\left[\int_a^t <<fn>>''(s)\,ds + <<fn>>'(a)\right]dt$.", "long"),
    (r"$\iint_D f(x,y)\,dx\,dy = \int_a^b\left[\int_{<<fn:A>>(x)}^{<<fn:B>>(x)} f(x,y)\,dy\right]dx$", "long"),
    (r"$\sum_{k=0}^{n} \binom{n}{k} a^k b^{n-k} = (a+b)^n,\quad \sum_{k=0}^n (-1)^k\binom{n}{k} = 0,\quad \sum_{k=0}^n \binom{n}{k} = 2^n$", "long"),
    (r"Условия Коши--Римана: $\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y},\quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$.", "long"),
    (r"$f^{(n)}(x) = \frac{n!}{2\pi i} \oint_C \frac{f(z)}{(z-x)^{n+1}}\,dz$", "long"),
    (r"Разложение в ряд Фурье: $f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty}\left[a_n \cos\frac{\pi n x}{L} + b_n \sin\frac{\pi n x}{L}\right]$", "long"),
    (r"$a_n = \frac{1}{L}\int_{-L}^{L} f(x)\cos\frac{\pi n x}{L}\,dx,\quad b_n = \frac{1}{L}\int_{-L}^{L} f(x)\sin\frac{\pi n x}{L}\,dx$", "long"),
    (r"Тогда $\left\|\sum_{k=1}^n <<gk>>_k e_k\right\|^2 = \sum_{k=1}^n |<<gk>>_k|^2$ (тождество Парсеваля).", "long"),

    # ── MULTI-FORMULA (several formulas per line, various spacings) ───────────
    # Trains the model to handle \quad / \, / \; / \qquad and natural spaces
    # between adjacent expressions — common in handwritten notes.
    (r"$<<fn:A>>(<<var>>) = <<var>>^<<exp>>$ \quad $<<fn:B>>(<<var>>) = \sin <<var>>$",  "mixed"),
    (r"$M_1(<<int:a>>;\, <<int:b>>)$ \quad $M_2(<<int:c>>;\, <<int:d>>)$",                "mixed"),
    (r"$a = <<int:a>>$, \quad $b = <<int:b>>$, \quad $c = <<int:c>>$",                    "mixed"),
    (r"$x_1 = <<int:a>>$,\; $x_2 = <<int:b>>$,\; $x_3 = <<int:c>>$",                      "mixed"),
    (r"$<<fn>>(<<var>>) = <<int:a>>$ при $<<var>> > 0$, \quad $<<fn>>(<<var>>) = <<int:b>>$ при $<<var>> < 0$", "mixed"),
    (r"$\lim_{n \to \infty} a_n = <<int:a>>$, \quad $\lim_{n \to \infty} b_n = <<int:b>>$", "mixed"),
    (r"$\int_0^1 <<var>>\,d<<var>> = \frac{1}{2}$ \qquad $\int_0^1 <<var>>^2\,d<<var>> = \frac{1}{3}$", "mixed"),
    (r"$<<fn>>'(<<var>>) = <<int>><<var>>$, \quad $<<fn>>''(<<var>>) = <<int>>$",         "mixed"),
    (r"$A = \{<<var>> \in \mathbb{R} : <<var>> > <<int:a>>\}$, \quad $B = \{<<var>> : <<var>> < <<int:b>>\}$", "mixed"),
    (r"$(<<int:a>>,\, <<int:b>>)$, \quad $(<<int:c>>,\, <<int:d>>)$, \quad $(<<int:e>>,\, <<int:f>>)$", "mixed"),
    (r"Решения: $<<var>>_1 = <<int:a>>$, \quad $<<var>>_2 = <<int:b>>$, \quad $<<var>>_3 = <<int:c>>$.", "mixed"),
    (r"$\sin <<gk>> = <<int:a>>$, \quad $\cos <<gk>> = <<int:b>>$, \quad $\mathrm{tg}\, <<gk>> = <<int:c>>$", "mixed"),
    (r"$x = <<int:a>>$, $y = <<int:b>>$ \quad $\Rightarrow$ \quad $z = <<int:c>>$",       "mixed"),
    (r"$<<int:a>> \leq <<var>> \leq <<int:b>>$, \quad $<<var>> \neq <<int:c>>$",          "mixed"),
    (r"$\vec{a} = (<<int:a>>,\, <<int:b>>,\, <<int:c>>)$, \qquad $\vec{b} = (<<int:d>>,\, <<int:e>>,\, <<int:f>>)$", "mixed"),
    (r"$<<gk:a>> = <<n:a>>^\circ$, \quad $<<gk:b>> = <<n:b>>^\circ$, \quad $<<gk:c>> = <<n:c>>^\circ$", "mixed"),
    (r"При $<<var>> = <<int:a>>$: $<<fn>>(<<var>>) = <<int:b>>$, \quad при $<<var>> = <<int:c>>$: $<<fn>>(<<var>>) = <<int:d>>$.", "mixed"),
    (r"$\det A = <<int:a>>$, \quad $\mathrm{tr}\, A = <<int:b>>$, \quad $\mathrm{rank}\, A = <<pos>>$", "mixed"),
]

_TEXT_TMPLS    = [(t, c) for t, c in TEMPLATES if c == "text"]
_FORMULA_TMPLS = [(t, c) for t, c in TEMPLATES if c == "formula"]
_MIXED_TMPLS   = [(t, c) for t, c in TEMPLATES if c == "mixed"]
_LONG_TMPLS    = [(t, c) for t, c in TEMPLATES if c == "long"]
_SHORT_TMPLS   = _TEXT_TMPLS + _FORMULA_TMPLS + _MIXED_TMPLS


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX rendering
# ──────────────────────────────────────────────────────────────────────────────

_FONTS = [
    ("cm",       ""),                                         # Computer Modern (default)
    ("lm",       r"\usepackage{lmodern}"),                   # Latin Modern
    ("times",    r"\usepackage{mathptmx}"),                  # Times + math
    ("palatino", r"\usepackage{mathpazo}"),                  # Palatino + math
    ("cmbright", r"\usepackage{cmbright}"),                  # CM Bright (sans)
]

# Default font sizes — overridden by config.synthetic_font_sizes at runtime
_FONTSIZES = [10, 11, 12, 14]

_DOC_TEMPLATE = r"""\documentclass[{size}pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T2A]{{fontenc}}
\usepackage[russian]{{babel}}
\usepackage{{amsmath,amssymb,amsthm}}
\usepackage{{geometry}}
\geometry{{paperwidth=55cm,paperheight=10cm,left=0.5cm,right=0.5cm,top=0.5cm,bottom=0.5cm}}
\pagestyle{{empty}}
\parindent=0pt
{pkg}
\begin{{document}}
\noindent {content}
\end{{document}}
"""


def _make_doc(content: str, font_pkg: str, size: int) -> str:
    return _DOC_TEMPLATE.format(size=size, pkg=font_pkg, content=content)


def _check_pdflatex() -> bool:
    try:
        subprocess.run(["pdflatex", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _render(content: str, font_pkg: str, size: int, dpi: int = 200) -> np.ndarray | None:
    """Compile LaTeX content to grayscale uint8 numpy array. Returns None on failure."""
    doc = _make_doc(content, font_pkg, size)
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "f.tex")
        pdf_path = os.path.join(tmpdir, "f.pdf")
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(doc)
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
                capture_output=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0 or not os.path.exists(pdf_path):
            return None
        try:
            pages = convert_from_path(pdf_path, dpi=dpi, grayscale=True)
        except Exception:
            return None
        if not pages:
            return None
        img = np.array(pages[0])
        img = crop_to_content(img)
        if img.size == 0:
            return None
        return img


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────

_CATEGORY_POOLS: dict[str, list] = {
    "text":    _TEXT_TMPLS,
    "formula": _FORMULA_TMPLS,
    "mixed":   _MIXED_TMPLS,
    "long":    _LONG_TMPLS,
}


def _pick_template(config: Config) -> tuple[str, str]:
    """Return (filled_content, category).

    Templates are sampled from each category pool weighted by synthetic_template_weights.
    Default weights are all 1.0 → uniform selection across the full template library.
    To double long examples: synthetic_template_weights={"long": 2.0, ...}.
    """
    w = config.synthetic_template_weights
    pairs = [(cat, pool) for cat, pool in _CATEGORY_POOLS.items() if pool]
    weights = [w.get(cat, 1.0) for cat, _ in pairs]
    (cat, pool), = random.choices(pairs, weights=weights, k=1)
    tmpl, _ = random.choice(pool)
    return _fill(tmpl), cat


def generate(config: Config, target_count: int, force: bool = False) -> None:
    out_dir    = config.synthetic_dir
    images_dir = os.path.join(out_dir, "images")
    labels_path = os.path.join(out_dir, "labels.json")
    meta_path   = os.path.join(out_dir, "meta.json")

    # Idempotency
    if not force and os.path.exists(labels_path):
        with open(labels_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Синтетика уже существует ({len(existing)} изображений). "
              f"Используйте --force для повторной генерации.")
        return

    if force and os.path.isdir(images_dir):
        import shutil
        shutil.rmtree(images_dir)
        print(f"Удалено: {images_dir}")
    for p in (labels_path, meta_path):
        if os.path.exists(p):
            os.remove(p)

    os.makedirs(images_dir, exist_ok=True)

    # Probe ALL fonts first, then pick up to synthetic_fonts_count at random.
    # This ensures equal chance for each font instead of always preferring the first N.
    print("Проверка доступных шрифтов...")
    all_ok_fonts: list[tuple[str, str]] = []
    for font_name, font_pkg in _FONTS:
        test_img = _render("test", font_pkg, 12, dpi=72)
        if test_img is not None:
            all_ok_fonts.append((font_name, font_pkg))
            print(f"  OK  {font_name}")
        else:
            print(f"  --  {font_name} (пропускаем)")

    if not all_ok_fonts:
        print("Ни один шрифт не работает. Проверьте установку LaTeX-пакетов.")
        sys.exit(1)

    k = min(config.synthetic_fonts_count, len(all_ok_fonts))
    available_fonts = random.sample(all_ok_fonts, k)
    print(f"Используем {k} шрифт(а) из {len(all_ok_fonts)} доступных: "
          f"{[n for n, _ in available_fonts]}")

    font_sizes = config.synthetic_font_sizes
    max_attempts = target_count * config.synthetic_max_attempts_ratio

    labels: dict[str, str] = {}
    n_ok = 0
    n_fail = 0
    cat_counts: dict[str, int] = {"text": 0, "formula": 0, "mixed": 0, "long": 0}

    pbar = tqdm(total=target_count, unit="img")
    attempts = 0

    while n_ok < target_count and attempts < max_attempts:
        attempts += 1
        content, category = _pick_template(config)

        if len(content) < config.synthetic_min_chars:
            n_fail += 1
            continue

        font_name, font_pkg = random.choice(available_fonts)
        size = random.choice(font_sizes)

        img = _render(content, font_pkg, size, dpi=config.synthetic_dpi)
        if img is None:
            n_fail += 1
            continue

        fname = f"{n_ok:06d}_{font_name}_{size}pt.png"
        fpath = os.path.join(images_dir, fname)

        from PIL import Image as PILImage
        PILImage.fromarray(img).save(fpath)

        labels[fname] = content
        cat_counts[category] = cat_counts.get(category, 0) + 1
        n_ok += 1
        pbar.update(1)

    pbar.close()

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    meta = {
        "generated_at":       datetime.now().isoformat(),
        "target_count":       target_count,
        "actual_count":       n_ok,
        "failed_renders":     n_fail,
        "fonts_used":         [n for n, _ in available_fonts],
        "font_sizes":         font_sizes,
        "dpi":                config.synthetic_dpi,
        "min_chars":          config.synthetic_min_chars,
        "template_weights":   config.synthetic_template_weights,
        "category_counts":    cat_counts,
        "template_counts": {
            "text":    len(_TEXT_TMPLS),
            "formula": len(_FORMULA_TMPLS),
            "mixed":   len(_MIXED_TMPLS),
            "long":    len(_LONG_TMPLS),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total = sum(cat_counts.values())
    print(f"\nГотово: {n_ok} изображений, {n_fail} сбоев рендера")
    print(f"Категории: " + "  ".join(
        f"{k}={v} ({100*v//max(total,1)}%)" for k, v in cat_counts.items()
    ))
    print(f"Следующий шаг: python prepare_data.py --datasets synthetic")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация синтетического датасета")
    parser.add_argument("--count",   type=int, default=None,
                        help="Количество изображений (по умолчанию из config)")
    parser.add_argument("--profile", default="rtx4060_8gb",
                        choices=["rtx4060_8gb", "rtx5090_32gb"],
                        help="GPU-профиль")
    parser.add_argument("--force",   action="store_true",
                        help="Перегенерировать с нуля")
    args = parser.parse_args()

    if not _PDF2IMAGE_OK:
        print("Ошибка: pdf2image не установлен. Установите: pip install pdf2image")
        print("Также нужен poppler: https://poppler.freedesktop.org/")
        sys.exit(1)

    if not _check_pdflatex():
        print("Ошибка: pdflatex не найден. Установите TeX Live или MiKTeX.")
        sys.exit(1)

    config = load_config(args.profile)
    target = args.count if args.count is not None else config.synthetic_count
    print(f"Профиль: {args.profile}  цель: {target} изображений")
    print(f"Выход: {os.path.abspath(config.synthetic_dir)}")
    print(f"Шаблонов: {len(TEMPLATES)} ({len(_LONG_TMPLS)} длинных, "
          f"long_ratio={config.synthetic_long_ratio})\n")

    generate(config, target_count=target, force=args.force)


if __name__ == "__main__":
    main()

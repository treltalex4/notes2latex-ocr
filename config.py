from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Config:
    # ===== GPU / Hardware =====
    device: str = "cuda"

    # ===== Model Architecture =====
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512          # верхний предел для позиционных индексов RoPE
    use_rope: bool = True           # RoPE в декодере вместо learnable PE

    # ===== CNN Backbone =====
    cnn_channels: tuple = (32, 64, 128, 256)
    target_height: int = 128
    max_width: int = 2048

    # ===== Training =====
    batch_size: int = 8
    grad_accum_steps: int = 4       # effective bs = batch_size * grad_accum_steps
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs_pretrain: int = 30       # этап 1: формулы (im2latex)
    epochs_mixed: int = 40          # этап 2: формулы + синтетика
    epochs_finetune: int = 20       # этап 3: свой датасет (с replay synthetic)
    warmup_steps: int = 1000
    patience: int = 7               # early stopping

    # ===== Mixed Precision =====
    use_amp: bool = True
    amp_dtype: str = "float16"      # "float16" | "bfloat16"

    # ===== Data Loading =====
    num_workers: int = 4
    prefetch_factor: int = 2

    # ===== Datasets =====
    datasets_stage1: list = field(default_factory=lambda: ["im2latex"])
    datasets_stage2: list = field(default_factory=lambda: ["im2latex", "synthetic"])
    dataset_weights_stage2: dict = field(
        default_factory=lambda: {"im2latex": 0.5, "synthetic": 0.5}
    )
    # Стадия 3: handwritten + replay synthetic против catastrophic forgetting
    dataset_weights_stage3: dict = field(
        default_factory=lambda: {"handwritten": 0.82, "synthetic": 0.18}
    )

    # ===== Dataset Quality =====
    # (см. Manual Tuning Guide в implementation_plan.md)
    synthetic_count: int = 40_000           # целевое число изображений
    latex_templates_count: int = 500        # кол-во уникальных LaTeX-шаблонов
    synthetic_fonts_count: int = 4          # сколько шрифтов выбрать из доступных
    synthetic_font_sizes: list = field(default_factory=lambda: [10, 11, 12, 14])  # pt

    # Шаблоны выбираются равновероятно из общего пула (80 шт.).
    # Естественное распределение: text≈25%, formula≈36%, mixed≈29%, long≈22%.
    # synthetic_template_weights — опциональные веса по категориям (None = равномерно)
    synthetic_template_weights: dict = field(
        default_factory=lambda: {"text": 1.0, "formula": 1.0, "mixed": 1.0, "long": 1.0}
    )

    synthetic_long_ratio: float = 0.15      # target fraction of long (>400 token) examples
    synthetic_dpi: int = 200                # разрешение рендера (DPI)
    synthetic_min_chars: int = 8            # пропускать контент короче N символов
    synthetic_max_attempts_ratio: int = 6   # max попыток рендера = count × коэффициент

    # ===== Augmentations =====
    # Per-dataset elastic factor: умножается на elastic_p из расписания.
    # 0.0 = elastic выключен полностью. 1.0 = расписание применяется как есть.
    # im2latex: типографские формулы — нужен мягкий шум. synthetic: рендер
    # LaTeX, должен "притворяться" рукописным — полная сила. handwritten:
    # рукопись уже elastic от природы, дополнительный шум вреден.
    elastic_factor_im2latex: float = 0.4
    elastic_factor_synthetic: float = 1.0
    elastic_factor_handwritten: float = 0.0
    augment_strength_max: float = 0.7       # верхний потолок не-elastic curriculum

    # Расписание elastic: list[(доля_эпох_стадии, p, alpha, sigma)]
    # доля от 0.0 до 1.0 — относительно числа эпох текущей стадии
    elastic_schedule_stage1: list = field(default_factory=lambda: [
        (0.33, 0.0,  0, 0),     # warmup: elastic выключен, учим чистую структуру
        (1.00, 0.2,  8, 4),     # introduce: mild
    ])
    elastic_schedule_stage2: list = field(default_factory=lambda: [
        (1.00, 0.5, 20, 5),     # full strength throughout — synthetic получает 1.0×, im2latex 0.4×
    ])
    elastic_schedule_stage3: list = field(default_factory=lambda: [
        (1.00, 0.3, 10, 5),     # только на replay synthetic
    ])

    # ===== Length Curriculum =====
    # Постепенное расширение допустимой длины формул (только stage 1)
    # list[(доля_эпох_стадии, max_tokens)]
    length_curriculum_stage1: list = field(default_factory=lambda: [
        (0.20, 200),    # первые 20% эпох — формулы до 200 токенов
        (0.50, 350),
        (1.00, 512),
    ])

    # ===== Tokenizer =====
    min_token_freq: int = 2
    tokenizer_max_len: int = 512

    # ===== Beam Search =====
    beam_size: int = 5
    beam_max_len: int = 600         # 2× от тренировочного max — запас на длинные строки
    length_penalty: float = 0.7

    # ===== Slicing (labeling/slicer.py) =====
    slice_max_width: int = 2500
    slice_deskew: bool = True

    slice_detect_dark_threshold: int = 40    # строгая маска: основа сегментации строк
    slice_expand_dark_threshold: int = 100   # мягкая маска: для захвата сирот
    slice_border_margin_px: int = 6

    slice_min_line_height: int = 12
    slice_min_line_width: int = 40

    slice_rlsa_h_kernel_factor: float = 1.0   # горизонтальное CLOSE = factor × median_h
    slice_rlsa_v_kernel_factor: float = 0.1  # малое вертикальное CLOSE для под/надстрочных
    slice_min_cc_area_for_scale: int = 25     # игнорировать шумовые CC при оценке масштаба

    slice_orphan_v_distance_factor: float = 4.0  # × median_h
    slice_orphan_h_tolerance_px: int = 40
    slice_orphan_min_area: int = 7
    slice_orphan_max_height_factor: float = 1.0   # не захватывать крупные CC (своя строка)

    slice_split_height_factor: float = 2.75    # × median_h; ниже — не делить
    slice_split_valley_ratio: float = 0.5    # долина < ratio × пик в профиле строки
    slice_split_min_run_factor: float = 0.10  # мин. длина долины × median_h

    slice_merge_y_overlap_ratio: float = 0.9

    slice_edge_touch_ratio: float = 0.04
    slice_edge_expand_x: int = 20
    slice_edge_expand_y: int = 14
    slice_max_edge_expand_iters: int = 5

    slice_pad_x: int = 20
    slice_pad_y: int = 14

    # ===== Paths =====
    data_dir: str = "data_raw"
    synthetic_dir: str = "data_synthetic"
    cache_dir: str = "data_cache"           # кэш предобработанных .npy тензоров
    checkpoint_dir: str = "checkpoints"
    my_dataset_dir: str = "my_dataset"
    plots_dir: str = "checkpoints/plots"


# ---------------------------------------------------------------------------
# GPU-профили
# ---------------------------------------------------------------------------

_PROFILES: dict[str, dict[str, Any]] = {
    "rtx4060_8gb": {
        # Все дефолтные значения Config уже настроены под RTX 4060 8GB
    },
    "rtx5090_32gb": {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 8,
        "num_decoder_layers": 8,
        "dim_feedforward": 2048,
        "max_seq_len": 1024,
        "batch_size": 32,
        "grad_accum_steps": 2,          # effective bs = 64
        "num_workers": 8,
        "amp_dtype": "bfloat16",
        "target_height": 160,           # больше пикселей → лучше мелкие индексы
        "max_width": 2048,
        "cnn_channels": (64, 128, 256, 512),
        "synthetic_count": 150_000,
        "latex_templates_count": 2000,
        "synthetic_fonts_count": 10,
        "synthetic_font_sizes": [10, 11, 12, 14, 16],  # добавляем 16pt для крупных формул
        "synthetic_dpi": 250,                           # выше DPI — чётче мелкие индексы
        "augment_strength_max": 1.0,
    },
}


def load_config(profile: str = "rtx4060_8gb", **overrides) -> Config:
    """Загружает GPU-профиль и применяет произвольные overrides.

    Использование:
        config = load_config()                              # RTX 4060 (дефолт)
        config = load_config("rtx5090_32gb")               # RTX 5090
        config = load_config("rtx4060_8gb", dropout=0.2)   # override
    """
    if profile not in _PROFILES:
        raise ValueError(
            f"Неизвестный профиль '{profile}'. Доступные: {list(_PROFILES)}"
        )

    params = dict(_PROFILES[profile])   # копия профиля (не мутируем глобальный dict)
    params.update(overrides)
    return Config(**params)

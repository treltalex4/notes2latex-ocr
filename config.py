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
    max_width: int = 1024

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
    synthetic_count: int = 40_000           # объём синтетики
    latex_templates_count: int = 500        # кол-во уникальных LaTeX-шаблонов
    synthetic_fonts_count: int = 4          # вариаций шрифтов при рендере
    synthetic_long_ratio: float = 0.15      # доля длинных (>400 токенов) примеров

    # ===== Augmentations =====
    use_elastic_im2latex: bool = True       # mild elastic на im2latex
    use_elastic_synthetic: bool = True      # medium elastic на synthetic
    use_elastic_handwritten: bool = False   # на handwritten НЕ применяется
    augment_strength_max: float = 0.7       # верхний потолок не-elastic curriculum

    # Расписание elastic: list[(доля_эпох_стадии, p, alpha, sigma)]
    # доля от 0.0 до 1.0 — относительно числа эпох текущей стадии
    elastic_schedule_stage1: list = field(default_factory=lambda: [
        (0.33, 0.0,  0, 0),     # warmup: elastic выключен, учим чистую структуру
        (1.00, 0.2,  8, 4),     # introduce: mild
    ])
    elastic_schedule_stage2: list = field(default_factory=lambda: [
        (0.50, 0.4, 15, 5),     # bridge: full elastic on
        (1.00, 0.5, 20, 5),     # full strength
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
        "synthetic_long_ratio": 0.20,
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

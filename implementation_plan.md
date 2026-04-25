# Пошаговый план реализации notes2latex-ocr

---

## Шаг 0 — Конфигурационная система (`config.py`)

Центральный модуль, из которого **все** скрипты берут параметры.
Ни один гиперпараметр не хардкодится в коде обучения / модели / данных.

```python
from dataclasses import dataclass, field

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
    max_seq_len: int = 512       # верхний предел для позиционных индексов RoPE
    use_rope: bool = True        # RoPE в декодере вместо learnable PE

    # ===== CNN Backbone =====
    cnn_channels: tuple = (32, 64, 128, 256)
    target_height: int = 128
    max_width: int = 1024        # лимит ширины изображения

    # ===== Training =====
    batch_size: int = 8
    grad_accum_steps: int = 4    # effective bs = 8 * 4 = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs_pretrain: int = 30    # этап 1: формулы
    epochs_mixed: int = 40       # этап 2: формулы + синтетика (+ handwritten replay)
    epochs_finetune: int = 20    # этап 3: свой датасет (с replay synthetic)
    warmup_steps: int = 1000
    patience: int = 7            # early stopping

    # ===== Mixed Precision =====
    use_amp: bool = True
    amp_dtype: str = "float16"   # "float16" | "bfloat16"

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

    # ===== Dataset Quality (см. секцию "Manual Tuning Guide" ниже) =====
    synthetic_count: int = 40_000           # объём синтетики
    latex_templates_count: int = 500        # количество уникальных LaTeX-шаблонов
    synthetic_fonts_count: int = 4          # вариаций шрифтов в рендере
    synthetic_long_ratio: float = 0.15      # доля длинных (>400 токенов) примеров

    # ===== Augmentations =====
    use_elastic_im2latex: bool = True       # elastic на im2latex (mild)
    use_elastic_synthetic: bool = True      # elastic на synthetic (medium)
    use_elastic_handwritten: bool = False   # на handwritten НЕ применяется
    augment_strength_max: float = 0.7       # верхний потолок curriculum

    # Расписание elastic: (доля_эпох_стадии, p, alpha, sigma)
    # Применяется внутри каждой стадии, доля от 0.0 до 1.0
    elastic_schedule_stage1: list = field(default_factory=lambda: [
        (0.33, 0.0, 0,  0),      # warmup: чистые данные
        (1.00, 0.2, 8,  4),      # introduce: mild
    ])
    elastic_schedule_stage2: list = field(default_factory=lambda: [
        (0.50, 0.4, 15, 5),
        (1.00, 0.5, 20, 5),
    ])
    elastic_schedule_stage3: list = field(default_factory=lambda: [
        (1.00, 0.3, 10, 5),      # только на replay synthetic
    ])

    # ===== Length Curriculum =====
    # Постепенное расширение допустимой длины формул в обучении
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
    beam_max_len: int = 600         # 2× от тренировочного max — запас на длиннее
    length_penalty: float = 0.7

    # ===== Paths =====
    data_dir: str = "data_raw"
    synthetic_dir: str = "data_synthetic"
    cache_dir: str = "data_cache"           # кэш предобработанных тензоров
    checkpoint_dir: str = "checkpoints"
    my_dataset_dir: str = "my_dataset"
    plots_dir: str = "checkpoints/plots"
```

### GPU-профили

Два пресета. Переключение: `config = load_config("rtx5090_32gb")`.

| Параметр | RTX 4060 (8 GB) | RTX 5090 (32 GB) |
|---|---|---|
| `d_model` | 256 | 512 |
| `nhead` | 8 | 8 |
| `num_encoder_layers` | 4 | 8 |
| `num_decoder_layers` | 4 | 8 |
| `dim_feedforward` | 1024 | 2048 |
| `max_seq_len` | 512 | 1024 |
| `batch_size` | 8 | 32 |
| `grad_accum_steps` | 4 (eff. 32) | 2 (eff. 64) |
| `num_workers` | 4 | 8 |
| `amp_dtype` | float16 | bfloat16 |
| `target_height` | 128 | 160 |
| `max_width` | 1024 | 2048 |
| `cnn_channels` | (32,64,128,256) | (64,128,256,512) |
| `synthetic_count` | 40 000 | 150 000 |
| `latex_templates_count` | 500 | 2000 |
| `synthetic_fonts_count` | 4 | 10 |
| `synthetic_long_ratio` | 0.15 | 0.20 |
| `augment_strength_max` | 0.7 | 1.0 |
| Параметры модели (≈) | ~15–20 M | ~60–80 M |

Функция `load_config(profile: str, **overrides)` загружает профиль
и позволяет перезаписать любой параметр через kwargs или CLI-аргументы.

### Manual Tuning Guide (под фактическое VRAM-потребление)

Секция настройки качества датасета **отдельно** от GPU-профилей —
её нужно подкручивать вручную, опираясь на реальное потребление памяти
во время обучения (мониторинг через `nvidia-smi` или TensorBoard).

**Если VRAM свободна (< 80% использования):** можно поднимать качество.

| Параметр | Эффект на качество | Эффект на VRAM/время |
|---|---|---|
| `target_height` (128 → 160 → 192) | Лучше распознавание мелких индексов | VRAM ↑↑, время эпохи ↑↑ |
| `max_width` (1024 → 1536 → 2048) | Больше длинных строк помещается без сжатия | VRAM ↑ (батчами с длинными примерами) |
| `batch_size` × `grad_accum_steps` | Мало влияет (effective одинаков) | VRAM ↑ при росте `batch_size` |
| `synthetic_count` (40k → 80k → 150k) | Лучше обобщение на русский текст | Только время эпохи ↑ |
| `latex_templates_count` (500 → 2000) | Больше разнообразие синтетики | Только время генерации ↑ |
| `synthetic_fonts_count` (4 → 10) | Робастность к шрифтам | Только время генерации ↑ |
| `d_model` (256 → 384) | Существенно лучше качество | VRAM ↑↑↑ (квадратично) |
| `num_*_layers` (4 → 6) | Лучше для длинных последовательностей | VRAM ↑↑ |

**Если VRAM > 95% (OOM-риск):** сначала снижать `batch_size`,
потом `target_height`, потом архитектурные параметры.

**Порядок «безопасных» апгрейдов** (от безопасных к рискованным):
1. `synthetic_count`, `latex_templates_count`, `synthetic_fonts_count` —
   качество данных без штрафа на VRAM.
2. `target_height` — ощутимый прирост качества, умеренный штраф.
3. `max_width` — помогает только на длинных примерах.
4. Архитектура (`d_model`, слои) — самые большие изменения, делать в последнюю очередь.

---

## Шаг 1 — Структура проекта и зависимости

**Создать файлы и папки:**
```
notes2latex-ocr/
├── config.py
├── prepare_data.py             # NEW: предподготовка датасета
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── tokenizer.py
│   ├── preprocess.py
│   ├── synthetic.py
│   └── cache.py                # NEW: работа с кэшем
├── model/
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── rope.py                 # NEW: Rotary Position Embeddings
│   └── model.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── beam_search.py
│   ├── schedules.py            # NEW: elastic + length curriculum
│   └── visualization.py
├── labeling/
│   ├── __init__.py
│   ├── slicer.py
│   ├── auto_label.py
│   └── label_tool.py
├── train.py
├── evaluate.py
├── finetune.py
├── tune.py
├── generate_synthetic.py
├── app.py
├── frontend.py
├── requirements.txt
└── README.md
```

**requirements.txt:**
```
torch>=2.6.0
torchvision
numpy
matplotlib
tqdm
opencv-python
pillow
nltk
editdistance
albumentations
optuna
tensorboard
streamlit
fastapi
uvicorn[standard]
python-multipart
google-generativeai
jupyter
ipykernel
pdf2image
h5py                    # для кэша датасета (опционально)
```

---

## Шаг 1.5 — Предподготовка датасета (`prepare_data.py`)

**Критическая инфраструктура. Запускается ОДИН раз перед `train.py`.**

### Зачем нужно

В `__getitem__` каждого датасета на каждой эпохе выполняется детерминированная
часть pipeline'а: `load PNG → decode → crop → binarize → resize`. На 100k
изображений × 30 эпох это **3 миллиона** одинаковых операций. CPU становится
бутылочным горлом, GPU простаивает в ожидании батча.

**Решение:** один раз прогнать всё через детерминированный pipeline и
сохранить результат как массивы uint8 в `data_cache/`.

### Что делает скрипт

```python
def prepare_dataset(config: Config, dataset_name: str):
    """
    Args:
        dataset_name: "im2latex" | "synthetic" | "handwritten"
    """
    # 1. Найти все исходные изображения
    samples = list_raw_samples(dataset_name)

    # 2. Для каждого:
    for raw_path, formula in tqdm(samples):
        # Детерминированная часть preprocess
        img = load_image(raw_path)
        if img is None:
            log_skipped(raw_path, "broken_file")
            continue

        img = crop_to_content(img)
        img = binarize(img)
        img = resize_preserve_aspect(img, config.target_height, config.max_width)

        # Фильтрация
        if img.shape[1] / img.shape[0] > 30:   # патологическое отношение
            log_skipped(raw_path, "bad_aspect_ratio")
            continue
        if len(tokenize(formula)) > config.tokenizer_max_len:
            log_skipped(raw_path, "formula_too_long")
            continue

        # Сохранение uint8-массива
        cache_path = f"{config.cache_dir}/{dataset_name}/{hash(raw_path)}.npy"
        np.save(cache_path, img)
        write_manifest_entry(cache_path, formula, length=len(tokenize(formula)))

    # 3. Статистика
    print_statistics(...)   # распределение длин, ширин, количество отфильтрованных
```

### Структура кэша

```
data_cache/
├── im2latex/
│   ├── manifest.json          # [{path, formula, length, width}, ...]
│   ├── stats.json              # min/max/mean/p95 длин и ширин
│   ├── skipped.log             # причины фильтрации
│   └── *.npy                   # uint8-массивы [H, W]
├── synthetic/
│   └── ...
└── handwritten/
    └── ...
```

### Что даёт

- **Ускорение эпохи в 5–15 раз** (зависит от диска).
- **Точная статистика** распределения длин — на основе которой можно
  обоснованно выставить `max_seq_len`, `tokenizer_max_len`, `max_width`
  в config (вместо угадывания).
- **Превентивная фильтрация** битых файлов — обучение не падает посреди эпохи.
- **Воспроизводимость** — все, кто работает с проектом, имеют идентичный
  отфильтрованный датасет.

### Размер на диске (оценка)

| Датасет | Объём | Размер |
|---|---|---|
| im2latex (100k × 128×600 uint8) | 100k | ~7.5 GB |
| synthetic (40k × 128×500 uint8) | 40k | ~2.5 GB |
| handwritten (~150 × 128×800 uint8) | 150 | ~15 MB |

**Итого:** ~10 GB на SSD. Норма для проекта такого размера.

### Запуск

```bash
python prepare_data.py --datasets im2latex synthetic handwritten
python prepare_data.py --datasets im2latex --force   # пересчитать с нуля
```

Скрипт идемпотентен — если кэш существует и manifest валиден, ничего
не пересчитывается.

---

## Шаг 2 — Подготовка датасетов

### 2a — im2latex-100k (печатные формулы → LaTeX)

1. Скачать архив с `zenodo.org/record/56198`
2. Распаковать в `data_raw/`

Структура:
```
data_raw/
├── formula_images/       # PNG изображения формул
├── im2latex_formulas.lst # LaTeX-код (строка = формула)
├── im2latex_train.lst    # индексы train
├── im2latex_validate.lst # индексы val
└── im2latex_test.lst     # индексы test
```

~100 000 пар. Используется на этапе 1 (pretrain) и этапе 2 (mixed).

### 2b — Синтетический датасет (русский текст + формулы → LaTeX)

Скрипт `generate_synthetic.py` + модуль `data/synthetic.py`.

**Pipeline генерации:**

1. **Подготовить LaTeX-шаблоны** — собрать библиотеку фрагментов в трёх
   категориях (объём = `config.latex_templates_count`):
   - Чисто текстовые: «Пусть дана функция...», «Доказательство.», «Теорема 1.»
   - Чисто формульные: `$\int_0^1 f(x)\,dx$`, `$\sum_{n=1}^{\infty} a_n$`
   - Смешанные: «Тогда $f'(x) = 2x$ при $x > 0$.»
   - Шаблоны сохраняются в `data/latex_templates/` (`.txt` файлы,
     одна строка = один пример)

2. **Длинные примеры (анти-проблема обобщения по длине):**

   Минимум `config.synthetic_long_ratio` (15–20%) шаблонов **обязательно**
   должны иметь длину > 400 токенов:
   - длинные формулы со многими подвыражениями: матрицы, системы уравнений,
     длинные суммы и интегралы
   - длинные предложения с 3+ формулами
   - сложные выражения с глубокой вложенностью

   Без этого хвост распределения недопредставлен в обучении и модель
   плохо работает на длинных строках в проде.

3. **Рендеринг (LaTeX → изображение):**
   ```
   Для каждого шаблона:
     1. Обернуть в минимальный .tex документ (\documentclass, babel russian, utf8)
     2. Скомпилировать pdflatex → PDF
     3. Растеризовать PDF → PNG (через pdf2image / Poppler)
     4. Обрезать по контенту (crop_to_content)
     5. Сохранить пару (image, latex_source)
   ```

4. **Вариативность шрифтов** (объём = `config.synthetic_fonts_count`):
   - Стандартный Computer Modern
   - С засечками (Times-like: `mathptmx`)
   - Sans-serif (`helvet`)
   - Рукописные шрифты (если доступны: `calligra`, `aurical`)
   - Разный размер шрифта (10pt – 14pt)

5. **Аугментации «под почерк»** — применяются НЕ при генерации, а в
   `preprocess.py` динамически (см. Шаг 4).

6. **Целевой объём:**

   | GPU-профиль | `synthetic_count` | `latex_templates_count` | `synthetic_fonts_count` |
   |---|---|---|---|
   | RTX 4060 (8 GB) | 30 000 – 50 000 | ~500 | 4 |
   | RTX 5090 (32 GB) | 100 000 – 200 000 | ~2000 | 10 |

   Все три параметра задаются в `config.py`.

7. **Формат хранения:**
   ```
   data_synthetic/
   ├── images/           # PNG файлы (исходники, до preprocess)
   ├── labels.json       # {filename: latex_string, ...}
   └── meta.json         # параметры генерации (фонты, шаблоны, дата)
   ```

   После генерации обязательно прогнать `prepare_data.py --datasets synthetic`
   для построения кэша.

### 2c — CROHME (рукописные формулы, опционально)

Датасет из соревнований ICDAR по распознаванию рукописных формул.
~10 000 примеров с InkML-разметкой, конвертируемой в LaTeX.

Если доступен — добавляется как дополнительный источник на этапе 2.
Датасет загружается и конвертируется скриптом `prepare_crohme.py`.

### 2d — Собственный рукописный датасет

≥ 150 размеченных строк из собственных конспектов.
Подробнее — шаг 14.

---

## Шаг 3 — Унифицированный токенизатор (`data/tokenizer.py`)

**Что обновить в текущей реализации:**

1. **Добавить кириллицу** — обновить regex-паттерн:
   ```python
   _TOKENIZE_PATTERN = re.compile(
       r"(\\[a-zA-Z]+)"        # LaTeX-команды: \frac, \text, ...
       r"|(\\[^a-zA-Z])"       # Escaped символы: \{, \\, ...
       r"|(\s)"                 # Пробел (критично для русского текста!)
       r"|([^\s])"             # Любой символ (Cyrillic, Latin, цифры, скобки...)
   )
   ```

2. **Пробел как токен** — в текущей версии пробелы пропускаются.
   Для русского текста пробел обязателен (разделитель слов).

3. **Построение словаря из ВСЕХ датасетов:**
   ```python
   all_formulas = (
       formulas_im2latex
       + formulas_synthetic
       + formulas_crohme       # если есть
       + formulas_my_dataset    # если есть
   )
   tokenizer.build_vocab(all_formulas, min_freq=2)
   ```

4. **Ожидаемый размер словаря:** ~800–1500 токенов
   (706 из im2latex + ~66 кириллических букв + ~30 новых символов/команд)

**Проверки:**
- `decode(encode(formula)) == formula` для формул из im2latex
- `decode(encode(text)) == text` для русского текста из синтетики
- `decode(encode(mixed)) == mixed` для смешанных строк

---

## Шаг 4 — Предобработка изображений (`data/preprocess.py`)

Текущая реализация **частично рабочая**. Обновления:

1. **Параметры из config** — `target_h`, `max_w` берутся из `Config`,
   не хардкодятся.

2. **ElasticTransform применяется ВЕЗДЕ, но с разной силой:**

   | Источник | `use_elastic_*` | Целевая сила |
   |---|---|---|
   | im2latex | `use_elastic_im2latex=True` | mild (alpha=8–20) |
   | synthetic | `use_elastic_synthetic=True` | medium-strong (alpha=15–25) |
   | handwritten | `use_elastic_handwritten=False` | НЕ применяется (рукопись уже elastic от природы) |

   На handwritten elastic ломает символы (точки → чёрточки, минусы исчезают).

3. **Сигнатура с расписанием:**
   ```python
   def apply_augmentations(
       img: np.ndarray,
       dataset_type: str,        # "im2latex" | "synthetic" | "handwritten"
       elastic_p: float,         # из расписания (см. Шаг 12)
       elastic_alpha: int,
       elastic_sigma: int,
       strength: float,          # общая интенсивность (0..1) — для не-elastic аугментаций
   ) -> np.ndarray:
       ...
   ```

   - Если `elastic_p == 0` — elastic пропускается (warmup-фаза).
   - Не-elastic аугментации (erosion/dilation, шум, blur, Affine) масштабируются
     через `strength`.

4. **Полный набор аугментаций:**
   - ElasticTransform (по флагу датасета и расписанию)
   - Erosion / Dilation (имитация толщины чернил)
   - Gaussian Noise (шум камеры)
   - GaussianBlur (расфокус)
   - RandomBrightnessContrast
   - Affine: поворот ±2°, scale ±5%

   На validation/test **НИКАКИХ аугментаций** — иначе метрики становятся
   нечестными.

---

## Шаг 5 — Dataset + Multi-dataset DataLoader (`data/dataset.py`)

### 5.0 — Чтение из кэша вместо raw PNG

Все датасеты читают `.npy`-файлы из `data_cache/<dataset_name>/`,
а не PNG из `data_raw/`. В `__getitem__` остаётся только:

```
load .npy → augment → to_tensor
```

Если кэш отсутствует — понятное сообщение об ошибке:
*«Кэш не найден. Запустите: `python prepare_data.py --datasets im2latex`»*

### 5a — Im2LatexDataset (обновление)

Сохраняется текущая логика, но `__getitem__` читает из кэша.
Добавить параметры расписания (передаются извне через атрибут):

```python
class Im2LatexDataset(Dataset):
    def __init__(self, ..., config: Config):
        ...
        self.elastic_p = 0.0          # обновляется train.py перед каждой эпохой
        self.elastic_alpha = 0
        self.elastic_sigma = 0
        self.strength = 0.0
        self.max_length = config.tokenizer_max_len   # обновляется длинным curriculum

    def __getitem__(self, idx):
        cached_path, formula = self.samples[idx]
        img = np.load(cached_path)
        img = apply_augmentations(
            img, dataset_type="im2latex",
            elastic_p=self.elastic_p, elastic_alpha=self.elastic_alpha,
            elastic_sigma=self.elastic_sigma, strength=self.strength,
        )
        return to_tensor(img), formula
```

### 5b — SyntheticDataset, HandwrittenDataset

Аналогично, но `dataset_type="synthetic"` / `"handwritten"`.

### 5c — MultiDatasetLoader (3 стадии)

```python
def build_multi_dataloaders(config: Config, tokenizer, stage: int):
    """
    stage=1: только im2latex (pretrain)
    stage=2: im2latex + synthetic (+ CROHME) с WeightedRandomSampler
    stage=3: handwritten 80% + synthetic replay 20% (anti-forgetting)
    """
```

**Стадия 3 — replay buffer против catastrophic forgetting:**

```python
# stage=3:
hw_dataset = HandwrittenDataset(...)
synth_replay = SyntheticDataset(...)   # тот же синтетический датасет
combined = ConcatDataset([hw_dataset, synth_replay])

# Веса: каждый handwritten пример = 1.0, каждый synthetic = подобран так,
# чтобы соотношение в батче было config.dataset_weights_stage3
# (по умолчанию 82% handwritten / 18% synthetic).
weights = compute_weights(...)
sampler = WeightedRandomSampler(weights, num_samples=len(combined))
```

`BucketBatchSampler` и `CollateFunction` остаются общими — интерфейс
одинаковый: `(image_tensor, latex_string)`.

### 5d — Length Curriculum в семплере

`BucketBatchSampler` принимает атрибут `current_max_length`. На ранних
эпохах семплер фильтрует примеры длиннее этого порога. Значение
обновляется `train.py` по расписанию `length_curriculum_stage1`.

---

## Шаг 6 — Гибридный CNN-ViT Encoder (`model/encoder.py`)

### CNN Backbone (Asymmetric Downsampling)

Свёрточная сеть с агрессивным сжатием высоты и бережным сжатием ширины.

| Профиль | Вход | Выход CNN | Длина seq |
|---|---|---|---|
| RTX 4060 | `[B, 1, 128, W]` | `[B, 256, 1, W/8]` | `W/8` |
| RTX 5090 | `[B, 1, 160, W]` | `[B, 512, 1, W/8]` | `W/8` |

Архитектура CNN:
```
Conv2d(1, 32, 3, padding=1) → BN → ReLU → MaxPool2d(2,2)     # H/2, W/2
Conv2d(32, 64, 3, padding=1) → BN → ReLU → MaxPool2d(2,2)    # H/4, W/4
Conv2d(64, 128, 3, padding=1) → BN → ReLU → MaxPool2d(2,1)   # H/8, W/4
Conv2d(128, 256, 3, padding=1) → BN → ReLU → MaxPool2d(2,1)  # H/16, W/4
Conv2d(256, d_model, 3, padding=1) → BN → ReLU → MaxPool2d(2,1)  # H/32, W/4
AdaptiveAvgPool2d((1, None))                                   # H → 1, W/4
```

Кол-во каналов на каждом слое = `config.cnn_channels`.

### Positional Encoding (encoder side)

**Sinusoidal 1D** (Sine-Cosine) — генерируется на лету для любого `seq_len`,
поддерживает длины, не виденные в обучении.

```python
class SinusoidalPE(nn.Module):
    def forward(self, x):
        # x: [B, seq_len, d_model]
        # Генерирует PE на лету для любого seq_len
```

### Transformer Encoder

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=config.d_model,
    nhead=config.nhead,
    dim_feedforward=config.dim_feedforward,
    dropout=config.dropout,
    batch_first=True,
)
self.transformer = nn.TransformerEncoder(
    encoder_layer,
    num_layers=config.num_encoder_layers,
    norm=nn.LayerNorm(config.d_model),
)
```

---

## Шаг 7 — Transformer Decoder (`model/decoder.py`)

### КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: RoPE вместо learnable PE

**Проблема learnable PE:** обучаемый `nn.Embedding(max_seq_len, d_model)`
**не обобщается** на длины больше тренировочного `max_seq_len`. Если в проде
встретится строка из 700 токенов — модель сломается на decoder side.

**Решение:** Rotary Position Embeddings (RoPE) — стандарт для современных
LLM (Llama, Mistral, Qwen). RoPE применяется **внутри attention** к
queries и keys, не требует фиксированного `max_seq_len` и хорошо
экстраполирует на длины, не виденные в обучении.

### Реализация (`model/rope.py`):

```python
class RotaryEmbedding(nn.Module):
    """RoPE для self-attention в декодере."""
    def __init__(self, dim: int, base: int = 10000):
        ...

    def forward(self, q, k):
        """
        Применяет ротацию к q и k. dim = d_model // nhead (per-head).
        Возвращает (q_rotated, k_rotated).
        """
```

### Компоненты декодера:

- **Token Embedding:** `nn.Embedding(vocab_size, d_model, padding_idx=0)`
- **Positional Encoding:** **RoPE** (применяется внутри self-attention),
  никакого добавления PE к эмбеддингу.
- **Decoder Layer:** ручная реализация с `F.scaled_dot_product_attention`:
  ```python
  class DecoderLayer(nn.Module):
      def __init__(self, d_model, nhead, dim_feedforward, dropout):
          self.self_attn = RoPESelfAttention(d_model, nhead, dropout)
          self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
          self.ffn = nn.Sequential(
              nn.Linear(d_model, dim_feedforward),
              nn.GELU(),
              nn.Dropout(dropout),
              nn.Linear(dim_feedforward, d_model),
          )
          self.norm1 = nn.LayerNorm(d_model)
          self.norm2 = nn.LayerNorm(d_model)
          self.norm3 = nn.LayerNorm(d_model)
  ```

  В `RoPESelfAttention.forward()` после линейных проекций q, k мы применяем
  RoPE-ротацию, потом вызываем `F.scaled_dot_product_attention` (FlashAttention).

- **Output Projection:** `nn.Linear(d_model, vocab_size)`

**Проверка длинной экстраполяции:**
- Обучение на длинах ≤ 512.
- Инференс на длинах 600, 800, 1000 — должно работать без деградации
  (опционально проверить на специально подобранных длинных синтетических примерах).

---

## Шаг 8 — Полная модель (`model/model.py`)

```python
class Notes2LaTeX(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        self.encoder = HybridEncoder(config)
        self.decoder = LaTeXDecoder(config, vocab_size)

    def forward(self, images, tgt):
        memory = self.encoder(images)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
        logits = self.decoder(tgt, memory, tgt_mask)
        return logits
```

Функция `count_parameters(model)` — подсчёт параметров.

**Ожидаемое количество параметров:**

| Профиль | Параметры |
|---|---|
| RTX 4060 (d=256, 4+4 слоя) | ~15–20 M |
| RTX 5090 (d=512, 8+8 слоёв) | ~60–80 M |

---

## Шаг 9 — Метрики (`utils/metrics.py`)

Реализовать:

- `bleu_score(predictions, references)` — BLEU-4 через `nltk`
- `exact_match(predictions, references)` — доля полных совпадений
- `token_accuracy(logits, targets, pad_idx=0)` — точность по токенам
- `character_error_rate(predictions, references)` — CER через `editdistance`
- `edit_distance_score(predictions, references)` — нормализованный Levenshtein

**Целевые значения:**

| Метрика | Цель (свой датасет) |
|---|---|
| BLEU-4 | > 0.70 |
| ExactMatch | ≥ 45% |
| CER | < 15% |

---

## Шаг 10 — Beam Search (`utils/beam_search.py`)

Функция `beam_search(model, image, tokenizer, config)`:

1. Прогнать изображение через encoder → memory
2. Начать с `[<SOS>]`
3. На каждом шаге расширить топ-`config.beam_size` гипотез
4. Остановка при `<EOS>` или `config.beam_max_len`
5. Вернуть лучшую гипотезу (по `score / length^config.length_penalty`)

Параметры по умолчанию:
- `beam_size = 5`
- `beam_max_len = 600` (2× от тренировочного — запас на длинные строки)
- `length_penalty = 0.7`

---

## Шаг 11 — Визуализация (`utils/visualization.py`)

- `plot_learning_curves(history, save_dir)` — графики loss, BLEU, CER по эпохам
  для каждого этапа обучения
- `show_predictions(model, dataset, tokenizer, n=10)` — изображения
  с предсказаниями (отдельно для формул, текста, смешанных)
- Сохранение в `config.plots_dir`

---

## Шаг 11.5 — Расписания (`utils/schedules.py`)

Утилиты для управления curriculum-параметрами по эпохам.

```python
def get_elastic_params(epoch: int, total_epochs: int, schedule: list) -> tuple:
    """
    schedule = [(threshold_ratio, p, alpha, sigma), ...]
    Возвращает (p, alpha, sigma) для текущей эпохи.
    """
    ratio = epoch / total_epochs
    for threshold, p, alpha, sigma in schedule:
        if ratio <= threshold:
            return p, alpha, sigma
    return schedule[-1][1:]


def get_max_length(epoch: int, total_epochs: int, schedule: list) -> int:
    """Возвращает текущий лимит длины формул из length_curriculum."""
    ratio = epoch / total_epochs
    for threshold, max_len in schedule:
        if ratio <= threshold:
            return max_len
    return schedule[-1][1]


def get_augment_strength(epoch: int, total_epochs: int, max_strength: float) -> float:
    """Линейный ramp-up интенсивности не-elastic аугментаций."""
    return min(max_strength, max_strength * (epoch / total_epochs))
```

Эти функции вызываются в `train.py` перед каждой эпохой и пробрасывают
параметры в датасеты через атрибуты (`dataset.elastic_p = ...`).

---

## Шаг 12 — Гибридное многоэтапное обучение (`train.py`)

### Стратегия (anti-forgetting)

**Стадия 1 (pretrain):** только im2latex. Модель учит каноническую LaTeX-структуру
на чистых данных. Elastic подключается во второй половине стадии (curriculum).

**Стадия 2 (mixed):** im2latex + synthetic (+ CROHME) одновременно через
`WeightedRandomSampler`. **В каждом батче** есть представители обоих типов —
это и есть anti-forgetting механизм. Модель никогда «не уходит» из домена im2latex.

**Стадия 3 (finetune):** handwritten 82% + replay synthetic 18%. Чисто
handwritten без replay привёл бы к забыванию синтетики. На handwritten
elastic ВЫКЛЮЧЕН (рукопись и так elastic). На replay synthetic elastic
включён.

### Общая структура

```python
def train(config: Config):
    tokenizer = build_unified_tokenizer(config)
    model = Notes2LaTeX(config, tokenizer.vocab_size)
    optimizer = AdamW(model.parameters(),
                      lr=config.learning_rate,
                      weight_decay=config.weight_decay)
    criterion = CrossEntropyLoss(ignore_index=PAD_ID)
    scaler = torch.amp.GradScaler(enabled=config.use_amp)

    # === Этап 1: Pretrain на im2latex ===
    train_loader, val_loader = build_multi_dataloaders(config, tokenizer, stage=1)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs_pretrain)
    train_loop(model, train_loader, val_loader, optimizer, scheduler,
               criterion, scaler, config.epochs_pretrain, config,
               stage_name="pretrain", stage_id=1)

    # === Этап 2: Mixed (im2latex + synthetic) ===
    train_loader, val_loader = build_multi_dataloaders(config, tokenizer, stage=2)
    optimizer = AdamW(model.parameters(),
                      lr=config.learning_rate * 0.5,    # снижаем LR при смене этапа
                      weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs_mixed)
    train_loop(model, train_loader, val_loader, optimizer, scheduler,
               criterion, scaler, config.epochs_mixed, config,
               stage_name="mixed", stage_id=2)

    # === Этап 3 (finetune.py) — отдельным скриптом ===
```

### Цикл обучения `train_loop()`

```python
def train_loop(model, train_loader, val_loader, optimizer, scheduler,
               criterion, scaler, total_epochs, config, stage_name, stage_id):

    elastic_schedule = {
        1: config.elastic_schedule_stage1,
        2: config.elastic_schedule_stage2,
        3: config.elastic_schedule_stage3,
    }[stage_id]

    for epoch in range(total_epochs):
        # 1. Получить curriculum-параметры на эту эпоху
        p, alpha, sigma = get_elastic_params(epoch, total_epochs, elastic_schedule)
        strength = get_augment_strength(epoch, total_epochs, config.augment_strength_max)

        # 2. Обновить параметры в датасетах
        for ds in train_loader.dataset.datasets:   # ConcatDataset
            ds.elastic_p = p if should_apply_elastic(ds.dataset_type, config) else 0.0
            ds.elastic_alpha = alpha
            ds.elastic_sigma = sigma
            ds.strength = strength

        # 3. Length curriculum (только в stage 1)
        if stage_id == 1:
            current_max = get_max_length(epoch, total_epochs,
                                          config.length_curriculum_stage1)
            train_loader.batch_sampler.current_max_length = current_max

        # 4. train_epoch()
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                  scaler, config)

        # 5. validate() — БЕЗ аугментаций (val_loader всегда чистый)
        val_loss, val_bleu, val_cer = validate(model, val_loader, criterion,
                                                tokenizer, config)

        # 6. scheduler.step()
        scheduler.step()

        # 7. Checkpoint + early stopping + TensorBoard
        ...


def should_apply_elastic(dataset_type: str, config: Config) -> bool:
    return {
        "im2latex": config.use_elastic_im2latex,
        "synthetic": config.use_elastic_synthetic,
        "handwritten": config.use_elastic_handwritten,
    }[dataset_type]
```

### Детали `train_epoch()`

```
- forward pass с torch.amp.autocast(dtype=config.amp_dtype)
- loss.backward() через scaler
- gradient accumulation (config.grad_accum_steps шагов)
- scaler.step(optimizer), scaler.update()
- optimizer.zero_grad(set_to_none=True)
```

### Сохранение checkpoint

```python
torch.save({
    'epoch': epoch,
    'stage': stage_name,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_loss': val_loss,
    'val_bleu': val_bleu,
    'config': asdict(config),
    'vocab_size': tokenizer.vocab_size,
}, f'{config.checkpoint_dir}/best_{stage_name}.pth')
```

### Graceful shutdown

Перехват `Ctrl+C` (SIGINT) — сохранение текущего checkpoint перед выходом.

---

## Шаг 13 — Скрипт оценки (`evaluate.py`)

- Загрузить модель из checkpoint
- Прогнать test set с Beam Search (БЕЗ аугментаций)
- Посчитать все метрики (BLEU, ExactMatch, CER, Token Accuracy, Edit Distance)
- Вывести таблицу результатов
- Сохранить 20+ примеров предсказаний (формулы / текст / смешанные)
- Сравнить метрики по этапам (pretrain vs mixed vs finetune)

---

## Шаг 14 — Сбор своего датасета (параллельно с шагами 3–13)

### 14a — Нарезка строк (`labeling/slicer.py`)

Функция `slice_page_into_lines(image_path, output_dir)`:

1. Загрузить фото страницы и применить бинаризацию
2. Компенсация наклона текста (Deskewing) — через Hough Transform или
   минимизацию дисперсии проекции
3. Поиск строк (горизонтальная проекция + Connected Components
   для обработки наслоений)
4. Передача предложенных линий разреза в GUI для быстрой проверки
5. Вырезать и сохранить строки в `my_dataset/images/`

### 14b — Авторазметка (`labeling/auto_label.py`)

1. Взять все PNG из `my_dataset/images/`
2. Для каждого — отправить в Gemini API с промптом:
   *«Распознай рукописный текст и формулы. Верни LaTeX-код.
   Русский текст оберни в \text{...}. Формулы — в $...$. ...»*
3. Сохранить в `my_dataset/labels_draft.json`
4. Graceful shutdown (Ctrl+C) — прогресс не теряется

### 14c — GUI верификации (`labeling/label_tool.py`)

Tkinter-приложение:

- Показывает изображение строки
- Показывает предложенный LaTeX (можно редактировать)
- Живой рендер формулы (matplotlib) для визуальной проверки
- Кнопки: Сохранить (Enter), Пропустить (Space), Назад
- Сохраняет итог в `my_dataset/labels.json`
- Запоминает прогресс

**Минимальный объём:** ≥ 150 размеченных строк (текст + формулы + смешанные).

После разметки прогнать `prepare_data.py --datasets handwritten`.

---

## Шаг 15 — Fine-tuning (`finetune.py`)

**Стадия 3 = handwritten 82% + synthetic replay 18%** (anti-forgetting).

1. Загрузить лучший checkpoint из этапа 2 (`best_mixed.pth`)
2. Создать комбинированный загрузчик:
   ```python
   train_loader, val_loader = build_multi_dataloaders(config, tokenizer, stage=3)
   ```
   `HandwrittenDataset` (handwritten part) и `SyntheticDataset` (replay).
3. Разделить handwritten 80/20 train/val
4. Дообучить:
   - `lr = config.learning_rate * 0.1` (т.е. 3e-5)
   - `weight_decay = 0.01`
   - `epochs = config.epochs_finetune` (20)
   - Early stopping `patience = config.patience` (7)
   - Elastic: `use_elastic_handwritten=False`, `use_elastic_synthetic=True`
     (на replay)
5. Сохранить в `checkpoints/best_finetune.pth`
6. Сравнить метрики до и после fine-tuning (таблица)

---

## Шаг 16 — Поиск гиперпараметров (`tune.py`)

Optuna Bayesian Search по ключевым параметрам:

```python
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    d_model = trial.suggest_categorical("d_model", [192, 256, 320])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    n_layers = trial.suggest_int("num_layers", 3, 6)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    # ... обучить N эпох (только Stage 1, короткий прогон), вернуть val_loss
```

Запускается **до** финального обучения. Лучшие параметры сохраняются
в `checkpoints/best_hparams.json` и подставляются в `Config`.

---

## Шаг 17 — Финальное приложение (`app.py` + `frontend.py`)

### Бэкенд (`app.py`): FastAPI

```python
@app.post("/predict")
async def predict(file: UploadFile):
    """Принимает изображение одной строки, возвращает LaTeX."""
    image = load_and_preprocess(file)
    latex = beam_search(model, image, tokenizer, config)
    return {"latex": latex}

@app.post("/predict-page")
async def predict_page(file: UploadFile):
    """Принимает фото страницы, нарезает на строки, возвращает LaTeX-документ."""
    lines = slice_page_into_lines(file)
    results = [beam_search(model, line, tokenizer, config) for line in lines]
    document = assemble_latex_document(results)
    return {"latex": document, "lines": results}
```

Модель загружается из checkpoint при старте сервера.

### Фронтенд (`frontend.py`): Streamlit

- Загрузка JPG/PNG (одна или несколько страниц)
- Отправка на бэкенд
- Отображение: LaTeX-код + рендер формул
- Скачивание `.tex` файла

### Запуск

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
streamlit run frontend.py
```

---

## Шаг 18 — Оформление

- `README.md` — описание, установка, запуск, результаты
- Графики в `checkpoints/plots/`:
  - loss по эпохам (все 3 этапа)
  - BLEU, CER по эпохам
  - elastic_p и max_length по эпохам (визуализация curriculum)
- Таблица метрик по этапам (pretrain → mixed → finetune)
- 15+ примеров работы:
  - Формулы: изображение → предсказание → эталон
  - Русский текст: изображение → предсказание → эталон
  - Смешанные строки: изображение → предсказание → эталон

---

## Итоговый порядок выполнения

```
Шаг 0    → config.py (центральная конфигурация + GPU-профили + tuning guide)
Шаг 1    → Структура проекта + requirements.txt
Шаг 2a   → Скачать im2latex-100k
Шаг 3    → data/tokenizer.py (обновить: кириллица + пробел)
Шаг 2b   → generate_synthetic.py + data/synthetic.py (синтетический датасет)
Шаг 1.5  → prepare_data.py (предподготовка датасета — кэш)
Шаг 4    → data/preprocess.py (обновить: elastic per-dataset, curriculum)
Шаг 5    → data/dataset.py (читать из кэша, stage 3 с replay)
Шаг 11.5 → utils/schedules.py (curriculum-расписания)
Шаг 6    → model/encoder.py (CNN Backbone + ViT)
Шаг 7    → model/decoder.py (FlashAttention Decoder + RoPE)
Шаг 8    → model/model.py (Notes2LaTeX)
Шаг 9    → utils/metrics.py (+ CER, Edit Distance)
Шаг 10   → utils/beam_search.py
Шаг 11   → utils/visualization.py
Шаг 16   → tune.py (Optuna Search — подобрать гиперпараметры)
Шаг 12   → train.py (3-стадийное обучение: pretrain → mixed)
Шаг 13   → evaluate.py
            ↕ параллельно
Шаг 14   → labeling/ (сбор своего датасета) + prepare_data --datasets handwritten
            ↓ после шагов 13 + 14
Шаг 15   → finetune.py (handwritten + replay synthetic)
Шаг 17   → app.py + frontend.py (FastAPI + Streamlit)
Шаг 18   → Оформление
```

### Критический путь

```
config.py → tokenizer → prepare_data.py → dataset → model → train.py
```

Это минимум для первого запуска обучения.
Синтетический датасет и labeling развиваются параллельно.

### Зависимости между шагами

```
Шаг 0 (config)   ─────────────────────────────────────→ нужен везде
Шаг 2a (im2latex) ──→ Шаг 3 (tokenizer)
Шаг 2b (synthetic) ──→ Шаг 3 (tokenizer)
Шаг 3 (tokenizer) ──→ Шаг 1.5 (prepare_data) ──→ Шаг 5 (dataset)
Шаг 4 (preprocess) + Шаг 11.5 (schedules) ──→ Шаг 5 (dataset)
Шаг 6 (encoder) + Шаг 7 (decoder + RoPE) ──→ Шаг 8 (model) ──→ Шаг 12 (train)
Шаг 14 (labeling) ──→ prepare_data --datasets handwritten ──→ Шаг 15 (finetune)
Шаг 12 (train) ──→ Шаг 13 (evaluate) ──→ Шаг 15 (finetune)
Шаг 15 (finetune) ──→ Шаг 17 (app)
```

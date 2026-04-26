# notes2latex-ocr — Команды запуска

---

## generate_synthetic.py

Генерирует синтетический датасет: рендерит LaTeX-шаблоны (русский текст +
формулы) через pdflatex в PNG-изображения. Нужен для Stage 2 обучения,
так как im2latex не содержит кириллицы.

**Зависимости:** pdflatex (TeX Live или MiKTeX), pdf2image + poppler.

```bash
python generate_synthetic.py
python generate_synthetic.py --count 200
python generate_synthetic.py --count 50000 --profile rtx5090_32gb
python generate_synthetic.py --force
```

### Флаги

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--count N` | int | из config | Целевое количество изображений. Если не указан, берётся `config.synthetic_count` (40 000 для RTX 4060, 150 000 для RTX 5090). Используй малые значения (200–500) для быстрой проверки что рендеринг работает. |
| `--profile` | str | `rtx4060_8gb` | GPU-профиль конфига. Влияет на `synthetic_count`, `synthetic_fonts_count`, `synthetic_long_ratio`. Варианты: `rtx4060_8gb`, `rtx5090_32gb`. |
| `--force` | flag | выкл. | Перегенерировать с нуля, даже если `data_synthetic/labels.json` уже существует. Без этого флага скрипт идемпотентен: если данные есть — выходит сразу. |

### Что делает

1. Проверяет доступность pdflatex и pdf2image.
2. Проверяет каждый из 5 шрифтов (CM, LM, Times, Palatino, CM Bright) тестовым рендером; недоступные пропускает.
3. Для каждого из N изображений:
   - Выбирает случайный шаблон (80 шт., категории: текст / формула / смешанный / длинный).
   - Заполняет плейсхолдеры `<<fn>>`, `<<var>>`, `<<n>>` и др. случайными значениями.
   - Выбирает случайный шрифт и кегль (10/11/12/14pt).
   - Компилирует `.tex → PDF → PNG` через pdflatex + pdf2image.
   - Обрезает пустые поля (`crop_to_content`).
4. Сохраняет в `data_synthetic/`:
   - `images/` — PNG-файлы (сырые, до preprocess).
   - `labels.json` — `{filename: latex_string}`.
   - `meta.json` — статистика: шрифты, количество по категориям, дата.

### После генерации

```bash
python prepare_data.py --datasets synthetic
python test_pipeline.py   # раздел "5. Синтетический датасет" покажет примеры
```

### Типичные ошибки

- **pdflatex не найден** — установи TeX Live (Linux/Mac) или MiKTeX (Windows).
- **pdf2image не установлен** — `pip install pdf2image` + установи Poppler.
- **Много сбоев рендера** — некоторые шаблоны с `<<var>>` могут конфликтовать; скрипт их пропускает и пробует следующий (лимит попыток = count × 6).

---

## prepare_data.py

Предобработка датасетов: запускается **один раз** перед обучением. Прогоняет
все изображения через детерминированный pipeline (`crop → resize → binarize`),
сохраняет результат как uint8-массивы `.npy` в `data_cache/`. Ускоряет эпоху
обучения в 5–15 раз, так как в `__getitem__` остаётся только `load .npy → augment → to_tensor`.

```bash
python prepare_data.py --datasets im2latex
python prepare_data.py --datasets im2latex synthetic handwritten
python prepare_data.py --datasets im2latex --force
python prepare_data.py --datasets synthetic --profile rtx5090_32gb
```

### Флаги

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--datasets` | list | `im2latex` | Один или несколько датасетов через пробел. Допустимые значения: `im2latex`, `synthetic`, `handwritten`. Порядок не важен. Запускать только для тех датасетов, которые уже скачаны/сгенерированы. |
| `--force` | flag | выкл. | Пересчитать кэш с нуля. Без этого флага скрипт идемпотентен: если `manifest.json` для датасета уже существует — пропускает его. Нужен после изменения `target_height` или `max_width` в config. |
| `--profile` | str | `rtx4060_8gb` | GPU-профиль. Влияет на `target_height` и `max_width` — именно под эти размеры кэшируются изображения. Если сменил GPU — запусти с `--force`. Варианты: `rtx4060_8gb`, `rtx5090_32gb`. |

### Что делает

Для каждого изображения в датасете:
1. Загружает PNG в оттенках серого.
2. `crop_to_content` — обрезает пустые поля (порог 250, отступ 8px).
3. Фильтр: если соотношение ширина/высота > 30 — пропускает (патологические изображения).
4. `resize_preserve_aspect` — масштабирует к `target_height` (128px), максимум `max_width` (1024px).
5. `binarize` — адаптивная бинаризация (blockSize ∝ высоте изображения).
6. Фильтр: если формула длиннее `tokenizer_max_len` токенов — пропускает.
7. Сохраняет `.npy` (uint8, [H, W]) с именем из MD5-хэша пути.

Для `im2latex` читает split из `im2latex_train.lst`, `im2latex_validate.lst`, `im2latex_test.lst` и записывает поле `split` в manifest.

Сохраняет в `data_cache/<dataset>/`:
- `manifest.json` — `[{npy_path, formula, length, width, split}]`.
- `stats.json` — min/p50/p95/p99/max по длинам и ширинам (важно для настройки `tokenizer_max_len`).
- `skipped.log` — причина и путь для каждого пропущенного изображения.

### Оценка размера кэша

| Датасет | Объём | Размер на диске |
|---------|-------|-----------------|
| im2latex (100k изображений) | ~100k .npy | ~7–8 GB |
| synthetic (40k изображений) | ~40k .npy | ~2–3 GB |
| handwritten (~150 изображений) | ~150 .npy | ~15 MB |

### После запуска

Посмотри `data_cache/im2latex/stats.json` — если `p95` длин сильно меньше
текущего `tokenizer_max_len`, можно его снизить (меньше VRAM на паддинг).

---

## test_pipeline.py

Визуальная проверка всего pipeline'а без запуска обучения. Сохраняет
тестовые изображения в папку `test/`. Запускать после любых изменений
в `preprocess.py`, `tokenizer.py`, `dataset.py`.

```bash
python test_pipeline.py
```

Флагов нет — всё управляется двумя переменными в начале файла:

### Переменные в начале файла

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `N_IMAGES_IM2LATEX` | `10` | Сколько изображений `sample_NN.png` сохранить из im2latex. Каждое содержит исходное имя файла и LaTeX-формулу. |
| `N_IMAGES_SYNTHETIC` | `5` | Сколько изображений `synthetic_NN.png` сохранить из синтетики. Раздел пропускается если `data_synthetic/` не существует. |

### Что проверяет

| Раздел | Что делает |
|--------|-----------|
| **1. Токенизатор** | Строит словарь на 2000 формул из im2latex. Проверяет roundtrip `encode → decode` для LaTeX-строк (ожидается OK) и кириллицы (ожидается FAIL — нормально, пока нет синтетики). |
| **2. Предобработка** | Прогоняет N_IMAGES_IM2LATEX изображений через `crop → resize → binarize`. Считает сколько прошло без ошибок. |
| **3. Аугментации** | Сохраняет 5 изображений: оригинал, warmup, im2latex elastic (alpha=60), synthetic elastic (alpha=120), handwritten (только нон-elastic). Используй эти файлы чтобы визуально проверить что elastic не слишком агрессивен. |
| **4. DataLoader** | Строит загрузчик из сырых PNG (без кэша). Считает батчи, сохраняет sample_NN.png с именем файла и формулой. |
| **5. Синтетический датасет** | Показывает synthetic_NN.png из `data_synthetic/`. Пропускается если папка не существует. |

### Выходные файлы в `test/`

```
test/
├── aug_1_original.png            # исходное изображение после preprocess
├── aug_2_warmup.png              # strength=0.7, elastic=0
├── aug_3_im2latex.png            # elastic alpha=60, strength=1.0
├── aug_4_synthetic.png           # elastic alpha=120, strength=1.0
├── aug_5_handwritten.png         # elastic выключен, нон-elastic включён
├── sample_01.png ... sample_10.png      # изображения из im2latex
└── synthetic_01.png ... synthetic_05.png  # изображения из синтетики
```

---

## Профили GPU

Выбор профиля влияет на параметры данных и (в будущем) архитектуру модели.

```bash
python generate_synthetic.py --profile rtx5090_32gb
python prepare_data.py       --profile rtx5090_32gb --force
```

| Параметр | RTX 4060 (8 GB) | RTX 5090 (32 GB) |
|----------|-----------------|------------------|
| `target_height` | 128px | 160px |
| `max_width` | 1024px | 2048px |
| `batch_size` | 8 | 32 |
| `d_model` | 256 | 512 |
| `num_*_layers` | 4 | 8 |
| `synthetic_count` | 40 000 | 150 000 |
| Параметров модели | ~15–20M | ~60–80M |

> Если меняешь профиль после того как кэш уже построен — пересобери его:
> `python prepare_data.py --datasets im2latex synthetic --force`

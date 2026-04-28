"""YOLO training & inference configuration for the line-detection model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class YoloConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    model: str = "yolo26x.pt"
    """Base weights file, resolved from yolo_training/weights/.
    Alternatives: 'yolo26n.pt', 'yolo26s.pt', 'yolo26l.pt', 'yolo26x.pt'."""

    # ── Hardware ──────────────────────────────────────────────────────────────
    device: str = "0"
    """GPU index ('0'), multiple GPUs ('0,1'), or 'cpu'."""

    # ── Image size ────────────────────────────────────────────────────────────
    imgsz: int = 1024
    """Training and inference image size (pixels). Larger = slower but better
    on high-res pages; 1024 is a good default for A4 scans."""

    # ── Training schedule ─────────────────────────────────────────────────────
    epochs: int = 70
    """Total training epochs."""

    batch: int = 2
    """Batch size. Use -1 for auto-detect based on GPU memory."""

    patience: int = 20
    """Early stopping: stop if no improvement for this many epochs."""

    close_mosaic: int = 10
    """Disable mosaic augmentation for the last N epochs (stabilises training)."""

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer: str = "AdamW"
    """'SGD', 'AdamW', 'Adam', 'NAdam', 'RAdam', 'RMSProp', or 'auto'.
    AdamW is more stable than SGD for small custom datasets."""

    lr0: float = 0.001
    """Initial learning rate. Lower than COCO defaults because we fine-tune."""

    lrf: float = 0.01
    """Final LR as a fraction of lr0 (lr_final = lr0 * lrf)."""

    momentum: float = 0.937
    """SGD momentum / Adam beta1."""

    weight_decay: float = 0.0005
    """L2 regularisation — helps prevent overfitting on small datasets."""

    warmup_epochs: float = 3.0
    """Epochs over which LR is linearly warmed up from warmup_bias_lr to lr0."""

    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    cos_lr: bool = True
    """Cosine LR schedule — smooth decay; better than step for fine-tuning."""

    # ── Backbone freezing ─────────────────────────────────────────────────────
    freeze: int = 0
    """Freeze first N layers. 0 = train everything.
    Set to 10 to freeze the backbone and only train neck + head — useful for
    very small datasets or a first warm-up stage."""

    # ── Loss weights ──────────────────────────────────────────────────────────
    box: float = 7.5
    """Bounding-box regression loss weight."""

    cls: float = 0.5
    """Classification loss weight."""

    dfl: float = 1.5
    """Distribution Focal Loss weight."""

    label_smoothing: float = 0.1
    """Prevents overconfidence on small datasets (0 = off, typical: 0.0–0.1)."""

    # ── Augmentation ──────────────────────────────────────────────────────────
    degrees: float = 3.0
    """Random rotation ±degrees. Handwritten lines are slightly tilted."""

    translate: float = 0.1
    """Random translation ±fraction of image size."""

    scale: float = 0.3
    """Random scale ±fraction. Handles photos taken at different distances."""

    shear: float = 0.0
    """Random shear (degrees). Usually not needed for note pages."""

    perspective: float = 0.00005
    """Random perspective warp — small value simulates slight camera tilt."""

    flipud: float = 0.0
    """Vertical flip probability. Keep 0: flipped pages make no sense."""

    fliplr: float = 0.0
    """Horizontal flip probability. Keep 0: text reads left-to-right."""

    mosaic: float = 1.0
    """Mosaic augmentation probability: combines 4 pages into one training
    sample. Very helpful — increases effective dataset size."""

    mixup: float = 0.0
    """MixUp probability. Off by default (blends two pages — hard to learn)."""

    copy_paste: float = 0.0
    """Copy-paste augmentation. Off: line bboxes are contiguous."""

    hsv_h: float = 0.015
    """Hue jitter (ink colour varies slightly across pens)."""

    hsv_s: float = 0.3
    """Saturation jitter (handwritten notes have low saturation)."""

    hsv_v: float = 0.4
    """Value/brightness jitter (different lighting conditions when photographed)."""

    # ── Fine-tuning (--finetune flag) ─────────────────────────────────────────
    finetune_data_yaml: str = "f_data.yaml"
    """YAML file (in yolo_training/) pointing to the fine-tuning dataset
    (e.g. f_dataset/). Used instead of data.yaml when --finetune is set."""

    finetune_epochs: int = 30
    """Fewer epochs for fine-tuning — model is already trained."""

    finetune_lr0: float = 0.0001
    """10× lower than base lr0. Fine-tuning needs gentler updates to avoid
    forgetting what was learned in the first stage."""

    finetune_freeze: int = 0
    """Layers to freeze during fine-tuning. Set to 10 to freeze backbone."""

    finetune_close_mosaic: int = 5
    """Disable mosaic earlier in fine-tuning — fewer total epochs."""

    # ── Inference (test_model.py) ──────────────────────────────────────────────
    test_conf: float = 0.35
    """Confidence threshold for test_model.py predictions."""

    test_imgsz: int = 1024
    """Image size for inference (can differ from training imgsz)."""


# Default configuration used by both train_yolo.py and test_model.py.
cfg = YoloConfig()

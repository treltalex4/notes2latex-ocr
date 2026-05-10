import json
import os
import random

import matplotlib
matplotlib.use("Agg")   # headless backend: не требует X-сервера
import matplotlib.pyplot as plt
import torch

from data.tokenizer import EOS_ID, SOS_ID


def plot_learning_curves(history_or_path, save_dir: str) -> None:
    """Строит loss и accuracy/EM кривые по эпохам.

    history_or_path: либо dict (то что хранится в history_*.json), либо путь к файлу.
    save_dir: куда сохранить png'шки (обычно config.plots_dir).
    """
    if isinstance(history_or_path, str):
        with open(history_or_path, encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = history_or_path

    epochs_data = history.get("epochs", [])
    if not epochs_data:
        print("plot_learning_curves: history пустая, пропускаю.")
        return

    os.makedirs(save_dir, exist_ok=True)
    stage_name = history.get("stage_name", "unknown")

    epochs     = [e["epoch"] + 1 for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss   = [e["val_loss"]   for e in epochs_data]
    train_acc  = [e["train_acc"]  for e in epochs_data]
    val_acc    = [e["val_acc"]    for e in epochs_data]
    val_em     = [e["val_em"]     for e in epochs_data]

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train_loss", linewidth=2)
    ax.plot(epochs, val_loss,   label="val_loss",   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss curves — stage {stage_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(save_dir, f"loss_{stage_name}.png")
    fig.savefig(loss_path, dpi=100)
    plt.close(fig)

    # --- Accuracy + EM ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, label="train_acc", linewidth=2)
    ax.plot(epochs, val_acc,   label="val_acc",   linewidth=2)
    ax.plot(epochs, val_em,    label="val_em",    linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"Accuracy / Exact Match — stage {stage_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = os.path.join(save_dir, f"accuracy_{stage_name}.png")
    fig.savefig(acc_path, dpi=100)
    plt.close(fig)

    print(f"Saved plots:\n  {loss_path}\n  {acc_path}")


@torch.no_grad()
def show_predictions(model, dataset, tokenizer, device, n: int = 10,
                     save_path: str = "checkpoints/plots/predictions.png",
                     max_len: int = 200) -> None:
    """Рендерит n случайных примеров из датасета с GT и предсказанием модели.

    Использует greedy decode (для качественного просмотра достаточно).
    """
    model.eval()
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        img, formula = dataset[idx]                # img: [1, H, W]
        img_np = img.squeeze(0).cpu().numpy()

        # Greedy decode
        image_batch = img.unsqueeze(0).to(device)  # [1, 1, H, W]
        memory, memory_kpm = model.encoder(image_batch)
        generated = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
        for _ in range(max_len):
            logits = model.decoder(generated, memory, memory_key_padding_mask=memory_kpm)
            next_id = logits[0, -1].argmax().item()
            generated = torch.cat(
                [generated, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_id == EOS_ID:
                break
        predicted = tokenizer.decode(generated[0].tolist())

        ax.imshow(img_np, cmap="gray", aspect="auto")
        is_match = predicted == formula
        status = "OK" if is_match else "DIFF"
        ax.set_title(
            f"[{status}]  GT:   {formula}\n"
            f"         PRED: {predicted}",
            fontsize=8, loc="left", family="monospace",
        )
        ax.axis("off")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Saved predictions to {save_path}")

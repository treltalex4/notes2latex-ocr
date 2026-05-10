import torch
import torch.nn.functional as F

from data.tokenizer import EOS_ID, SOS_ID


@torch.no_grad()
def beam_search(model, image, tokenizer, config, src_key_padding_mask=None) -> str:
    """Beam search декодирование ОДНОГО изображения.

    image: [1, 1, H, W] — батч из одной картинки.
    src_key_padding_mask: [1, W] — паддинг на пиксельной ширине, опционально.

    Возвращает строку — лучшую гипотезу по нормированному score.
    """
    device = image.device
    K = config.beam_size
    max_len = config.beam_max_len
    alpha = config.length_penalty

    model.eval()

    # 1. Encoder один раз. memory: [1, src_len, d_model]
    memory, memory_kpm = model.encoder(image, src_key_padding_mask=src_key_padding_mask)

    # 2. Расширяем memory до K копий — для параллельного decode K beams.
    # expand создаёт view со страйдом 0, contiguous() копирует чтобы PyTorch
    # не жаловался при дальнейших операциях.
    memory_k = memory.expand(K, -1, -1).contiguous()
    memory_kpm_k = memory_kpm.expand(K, -1).contiguous() if memory_kpm is not None else None

    # 3. Начальное состояние: один beam = [SOS] со score 0.
    beams = torch.full((1, 1), SOS_ID, dtype=torch.long, device=device)
    beam_scores = torch.zeros(1, device=device)

    finished: list[tuple[float, list[int]]] = []  # (normalized_score, sequence)

    for _step in range(max_len - 1):
        cur_K = beams.shape[0]
        cur_mem = memory_k[:cur_K]
        cur_mem_kpm = memory_kpm_k[:cur_K] if memory_kpm_k is not None else None

        # Decoder: получаем logits на ВСЕХ позициях, нужна только последняя.
        logits = model.decoder(beams, cur_mem, memory_key_padding_mask=cur_mem_kpm)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)   # [cur_K, V]
        V = log_probs.shape[-1]

        # Объединённый score = старый + log_prob нового токена. Получаем
        # cur_K × V кандидатов. flat — линейная индексация по ним.
        combined = beam_scores.unsqueeze(1) + log_probs       # [cur_K, V]
        flat = combined.view(-1)                              # [cur_K * V]

        # Топ-K кандидатов: индекс i соответствует beam = i // V, token = i % V.
        top_scores, top_indices = flat.topk(K)
        beam_idx  = top_indices // V
        token_idx = top_indices %  V

        # Расширяем выбранные beams новыми токенами. beams[beam_idx]: [K, T].
        new_beams = torch.cat([beams[beam_idx], token_idx.unsqueeze(1)], dim=1)

        # Кандидаты с EOS уходят в finished, остальные продолжают.
        is_eos = token_idx == EOS_ID
        for i in range(K):
            if is_eos[i]:
                seq = new_beams[i].tolist()
                length = len(seq)
                # Length penalty: делим на length^alpha. alpha=0.7 → штрафует
                # короткие последовательности, не наказывая длинные слишком сильно.
                normalized = top_scores[i].item() / (length ** alpha)
                finished.append((normalized, seq))

        active = ~is_eos
        beams = new_beams[active]
        beam_scores = top_scores[active]

        # Стоп: либо нет активных beams, либо набрали достаточно завершённых.
        if beams.shape[0] == 0:
            break
        if len(finished) >= K:
            # Эвристика раннего выхода. Можно усилить: сравнить best finished с
            # лучшим возможным active. Для простоты — просто стоп.
            break

    # Если max_len достигнут с активными beams — добавляем их как кандидаты
    # (без EOS они "недоговорённые", но это всё что есть).
    for i in range(beams.shape[0]):
        seq = beams[i].tolist()
        length = max(len(seq), 1)
        normalized = beam_scores[i].item() / (length ** alpha)
        finished.append((normalized, seq))

    if not finished:
        return ""

    # Лучший по нормированному score.
    finished.sort(key=lambda x: x[0], reverse=True)
    best_seq = finished[0][1]
    return tokenizer.decode(best_seq)

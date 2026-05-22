import torch
import torch.nn.functional as F

from data.tokenizer import EOS_ID, SOS_ID


def _reorder_kv_caches(kv_caches, beam_idx):
    """Реордерит KV-кэши декодера по batch dim под новый порядок beams.

    kv_caches: opaque структура, возвращаемая decoder.forward_step. Обычно
               list/tuple из (k, v) тензоров по слоям, но может быть произвольно
               вложенной — рекурсивный обход покрывает все варианты.
    beam_idx: [K_new] — для каждого нового beam индекс источника в старом cur_K.
              Может содержать дубликаты (несколько топ-кандидатов из одного beam'а)
              — index_select корректно создаёт копии нужных слайсов.
    """
    if kv_caches is None:
        return None
    if isinstance(kv_caches, torch.Tensor):
        return kv_caches.index_select(0, beam_idx)
    if isinstance(kv_caches, dict):
        return {k: _reorder_kv_caches(v, beam_idx) for k, v in kv_caches.items()}
    if isinstance(kv_caches, (list, tuple)):
        return type(kv_caches)(_reorder_kv_caches(item, beam_idx) for item in kv_caches)
    return kv_caches


@torch.no_grad()
def beam_search(model, image, tokenizer, config, src_key_padding_mask=None) -> str:
    """Beam search декодирование ОДНОГО изображения с KV-cache.

    image: [1, 1, H, W] — батч из одной картинки.
    src_key_padding_mask: [1, W] — паддинг на пиксельной ширине, опционально.

    Возвращает строку — лучшую гипотезу по нормированному score (length-penalty α).

    Сложность с KV-cache: O(T × K) compute, O(T × K) memory.
    Без KV-cache было бы O(T² × K) — на T=600 это разница в ~200×.
    """
    device = image.device
    K = config.beam_size
    max_len = config.beam_max_len
    alpha = config.length_penalty

    model.eval()

    # 1. Encoder один раз. Используем compiled версию если есть (как в greedy_decode_batch).
    encoder = getattr(model, "_encoder_for_decode", None) or model.encoder
    memory, memory_kpm = encoder(image, src_key_padding_mask=src_key_padding_mask)

    # Расширяем memory до K копий — beam'ы шарят encoder output параллельно.
    # expand создаёт view со страйдом 0, contiguous() копирует для последующих ops.
    memory_k = memory.expand(K, -1, -1).contiguous()
    memory_kpm_k = memory_kpm.expand(K, -1).contiguous() if memory_kpm is not None else None

    # 2. Начальное состояние: один beam = [SOS] со score 0.
    # cur_token — последний токен (для forward_step), sequences — полная история
    # (для финальных decode).
    cur_token   = torch.full((1, 1), SOS_ID, dtype=torch.long, device=device)
    sequences   = torch.full((1, 1), SOS_ID, dtype=torch.long, device=device)
    beam_scores = torch.zeros(1, device=device)
    kv_caches   = None   # инициализируется после первого forward_step

    finished: list[tuple[float, list[int]]] = []   # (normalized_score, sequence)

    for step in range(max_len - 1):
        cur_K = cur_token.shape[0]
        cur_mem = memory_k[:cur_K]
        cur_mem_kpm = memory_kpm_k[:cur_K] if memory_kpm_k is not None else None

        # forward_step с KV-cache: Q считается только для нового токена,
        # K/V предыдущих позиций берутся из кэша. На первом шаге kv_caches=None →
        # декодер инициализирует кэш на SOS.
        logits, kv_caches = model.decoder.forward_step(
            cur_token, cur_mem,
            memory_key_padding_mask=cur_mem_kpm,
            kv_caches=kv_caches,
            position_offset=step,
        )
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)   # [cur_K, V]
        V = log_probs.shape[-1]

        # Combined score = текущий + log_prob нового токена. Получаем cur_K × V
        # кандидатов; flat — линейная индексация по ним.
        combined = beam_scores.unsqueeze(1) + log_probs       # [cur_K, V]
        flat = combined.view(-1)                              # [cur_K * V]

        top_scores, top_indices = flat.topk(K)
        beam_idx  = top_indices // V                          # [K] источник в cur_K
        token_idx = top_indices %  V                          # [K] выбранный токен

        # Полные новые последовательности — для записи в finished.
        new_sequences = torch.cat([sequences[beam_idx], token_idx.unsqueeze(1)], dim=1)

        # EOS-кандидаты переезжают в finished, остальные продолжают.
        is_eos = token_idx == EOS_ID
        if is_eos.any():
            eos_indices = is_eos.nonzero(as_tuple=True)[0].tolist()
            for i in eos_indices:
                seq = new_sequences[i].tolist()
                length = len(seq)
                # Length penalty: делим на length^alpha. alpha<1 штрафует
                # короткие, но не наказывает длинные слишком сильно (alpha=0.7
                # — google/baidu стандарт для NMT).
                normalized = top_scores[i].item() / (length ** alpha)
                finished.append((normalized, seq))

        active = ~is_eos
        if not active.any():
            break

        # Обновляем активные beams для следующего шага.
        sequences   = new_sequences[active]
        beam_scores = top_scores[active]
        cur_token   = token_idx[active].unsqueeze(1)

        # Реордерим KV-cache под источники активных beams. Без этого кэш слоёв
        # decoder остаётся в порядке old beams, и атеншн будет смотреть на
        # чужую историю токенов.
        kv_caches = _reorder_kv_caches(kv_caches, beam_idx[active])

        # Строгий критерий раннего выхода: если best finished score уже не хуже
        # теоретического верхнего предела любого active beam — продолжать бесполезно.
        #
        # Upper bound active beam:
        #   future_score  ≤ current_score      (extensions добавляют log_probs ≤ 0)
        #   future_length ≥ current_length + 1 (минимум один EOS впереди)
        #   Best normalized = current_max_score / (current_length + 1)^alpha
        #   — достигается если следующий токен EOS с log_prob = 0.
        if finished:
            best_finished = max(s for s, _ in finished)
            min_future_len = sequences.shape[1] + 1
            best_active_upper = beam_scores.max().item() / (min_future_len ** alpha)
            if best_finished >= best_active_upper:
                break

    # Если max_len достигнут с активными beams — добавляем их как кандидаты
    # (без EOS они "недоговорённые", но это всё что есть — лучше чем пустота).
    for i in range(sequences.shape[0]):
        seq = sequences[i].tolist()
        length = max(len(seq), 1)
        normalized = beam_scores[i].item() / (length ** alpha)
        finished.append((normalized, seq))

    if not finished:
        return ""

    finished.sort(key=lambda x: x[0], reverse=True)
    return tokenizer.decode(finished[0][1])


@torch.no_grad()
def beam_search_batch(model, images, src_key_padding_mask, tokenizer, config) -> list[str]:
    """Batched beam search для B изображений × K beams параллельно.

    images: [B, 1, H, W]
    src_key_padding_mask: [B, W] или None
    Возвращает: list из B строк.

    Внутренняя укладка beams — image-major, beam-minor: индекс i = b*K + k,
    где b — индекс изображения (0..B-1), k — индекс beam'а внутри изображения (0..K-1).
    Все тензоры состояния имеют batch dim = B*K.

    Сравнительно с single-image beam_search: ~5-10× быстрее на batch B≥16,
    т.к. GPU обрабатывает B*K beams одновременно вместо K в loop по картинкам.
    """
    device = images.device
    B = images.shape[0]
    K = config.beam_size
    max_len = config.beam_max_len
    alpha = config.length_penalty
    NEG_INF = float("-inf")

    model.eval()

    # 1. Encoder.
    encoder = getattr(model, "_encoder_for_decode", None) or model.encoder
    memory, memory_kpm = encoder(images, src_key_padding_mask=src_key_padding_mask)
    src_len = memory.shape[1]
    d_model = memory.shape[-1]

    # 2. STEP 0 special: 1 beam на картинку, B beams общим счётом.
    # Все стартуют с SOS — log_probs для всех одинаковые до forward'а, нет смысла
    # дублировать в B*K раньше времени.
    cur_token = torch.full((B, 1), SOS_ID, dtype=torch.long, device=device)
    logits, kv_caches = model.decoder.forward_step(
        cur_token, memory,
        memory_key_padding_mask=memory_kpm,
        kv_caches=None,
        position_offset=0,
    )
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)   # [B, V]
    V = log_probs.shape[-1]

    # Top-K первых токенов per image.
    top_scores, top_indices = log_probs.topk(K, dim=1)    # [B, K]

    # 3. Раздуваем до B*K: image-major (b*K + k).
    sos_col = torch.full((B * K, 1), SOS_ID, dtype=torch.long, device=device)
    first_tokens = top_indices.reshape(B * K, 1)
    sequences = torch.cat([sos_col, first_tokens], dim=1)  # [B*K, 2]
    cur_token = first_tokens
    beam_scores = top_scores.reshape(B * K)

    # Расширяем memory и kv_caches до B*K. Каждое изображение дублируется K раз.
    memory_bk = (memory.unsqueeze(1).expand(B, K, src_len, d_model)
                 .reshape(B * K, src_len, d_model).contiguous())
    memory_kpm_bk = None
    if memory_kpm is not None:
        memory_kpm_bk = (memory_kpm.unsqueeze(1).expand(B, K, src_len)
                         .reshape(B * K, src_len).contiguous())
    dup_idx = torch.arange(B, device=device).repeat_interleave(K)
    kv_caches = _reorder_kv_caches(kv_caches, dup_idx)

    # 4. Per-image finished list + EOS handling step 0.
    finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(B)]

    is_eos_0 = first_tokens.squeeze(1) == EOS_ID
    for gi in is_eos_0.nonzero(as_tuple=True)[0].tolist():
        b = gi // K
        seq = sequences[gi].tolist()
        normalized = beam_scores[gi].item() / (len(seq) ** alpha)
        finished[b].append((normalized, seq))
        beam_scores[gi] = NEG_INF   # «мёртвый» beam — больше не участвует в topk

    # 5. STEPS 1+: B*K beams в параллель.
    b_offsets = torch.arange(B, device=device).unsqueeze(1) * K   # [B, 1]

    for step in range(1, max_len - 1):
        # Все beams мертвы → выходим.
        if torch.isinf(beam_scores).all():
            break

        logits, kv_caches = model.decoder.forward_step(
            cur_token, memory_bk,
            memory_key_padding_mask=memory_kpm_bk,
            kv_caches=kv_caches,
            position_offset=step,
        )
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)   # [B*K, V]

        # Reshape для per-image topk: каждая картинка имеет свой K*V набор кандидатов.
        combined = beam_scores.view(B, K, 1) + log_probs.view(B, K, V)   # [B, K, V]
        flat = combined.view(B, K * V)
        top_scores, top_indices = flat.topk(K, dim=1)         # [B, K]

        beam_idx_local = top_indices // V                     # [B, K] in 0..K-1
        token_idx_bk   = top_indices %  V                     # [B, K]
        beam_idx_global = (b_offsets + beam_idx_local).reshape(B * K)   # [B*K] in 0..B*K

        # Новые последовательности через gather по beam_idx_global.
        new_sequences = torch.cat([
            sequences.index_select(0, beam_idx_global),
            token_idx_bk.reshape(B * K, 1),
        ], dim=1)

        new_scores = top_scores.reshape(B * K)
        token_idx_flat = token_idx_bk.reshape(B * K)

        # EOS-кандидаты → finished, их слоты помечаем -inf.
        is_eos = token_idx_flat == EOS_ID
        for gi in is_eos.nonzero(as_tuple=True)[0].tolist():
            b = gi // K
            seq = new_sequences[gi].tolist()
            normalized = new_scores[gi].item() / (len(seq) ** alpha)
            finished[b].append((normalized, seq))
            new_scores[gi] = NEG_INF

        # Обновляем state.
        sequences   = new_sequences
        beam_scores = new_scores
        cur_token   = token_idx_flat.unsqueeze(1)
        kv_caches   = _reorder_kv_caches(kv_caches, beam_idx_global)

        # Per-image early stopping: если best finished превзошёл верхний предел
        # активных beams этого изображения — все его слоты в -inf.
        cur_len = sequences.shape[1]
        min_future_len = cur_len + 1
        # Per-image max активного скора (NEG_INF если все мертвы).
        active_max = beam_scores.view(B, K).max(dim=1).values   # [B]
        upper_bound = active_max / (min_future_len ** alpha)
        # Best finished tensor (NEG_INF если ещё нет finished для картинки).
        for b in range(B):
            if not finished[b]:
                continue
            if torch.isinf(active_max[b]):
                continue   # уже все мёртвые, ничего убивать
            best_fin = max(s for s, _ in finished[b])
            if best_fin >= upper_bound[b].item():
                beam_scores[b * K:(b + 1) * K] = NEG_INF

    # 6. Добиваем активные beams в finished (max_len достигнут без EOS).
    for gi in range(B * K):
        score = beam_scores[gi].item()
        if score == NEG_INF:
            continue
        b = gi // K
        seq = sequences[gi].tolist()
        normalized = score / (max(len(seq), 1) ** alpha)
        finished[b].append((normalized, seq))

    # 7. Лучший по нормированному score per image.
    results: list[str] = []
    for b in range(B):
        if not finished[b]:
            results.append("")
        else:
            finished[b].sort(key=lambda x: x[0], reverse=True)
            results.append(tokenizer.decode(finished[b][0][1]))
    return results

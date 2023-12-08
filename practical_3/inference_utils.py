import torch
from validation.utils import generate_square_subsequent_mask, DEVICE
from validation.CustomTokenizer import CustomTokenizer

import torch.nn.functional as F

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break
    return ys


def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, beam_size):
    # Initialize the beam with the start symbol and an initial score
    initial_beam = (torch.tensor([start_symbol], device=DEVICE), 0.0)  # (sequence, score)
    beams = [initial_beam]

    for _ in range(max_len):
        new_beams = []
        for beam in beams:
            seq, score = beam
            if seq[-1] == end_symbol:
                new_beams.append(beam)
                continue

            prob = get_prob_from_model(model, seq, src, src_mask)

            topk_prob, topk_indices = torch.topk(prob, beam_size)

            for prob, word_idx in zip(topk_prob, topk_indices):
                new_seq = torch.cat([seq, word_idx.view(1).to(DEVICE)])
                new_score = score + torch.log(prob)
                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

    best_seq, _ = max(beams, key=lambda x: x[1])
    return best_seq


def top_k_sampling(logits, k):
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = -float('Inf')
    return torch.multinomial(F.softmax(logits, dim=-1), 1)

def top_p_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    large_negative_value = -1e10
    logits[indices_to_remove] = large_negative_value
    chosen_index = torch.multinomial(F.softmax(logits, dim=-1), 1)
    return chosen_index

def random_sampling(logits, temperature=1.0):
    logits = logits / temperature
    return torch.multinomial(F.softmax(logits, dim=-1), 1)

def advanced_sampling_decode(model, src, src_mask, max_len, tokenizer, sampling_method, **kwargs):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(tokenizer.BOS_IDX).type(torch.long).to(DEVICE)

    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        logits = model.generator(out[:, -1])
        next_word = sampling_method(logits, **kwargs).item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == tokenizer.EOS_IDX:
            break

    return ys


def get_prob_from_model(model, seq, src, src_mask):
    model.eval()

    memory = model.encode(src, src_mask)
    seq = seq.unsqueeze(1)

    tgt_mask = generate_square_subsequent_mask(seq.size(0)).to(DEVICE)

    output = model.decode(seq, memory, tgt_mask)

    prob = model.generator(output[-1])

    return torch.softmax(prob, dim=-1).squeeze(0)


def summarize(
    model: torch.nn.Module,
    src_sentence: str,
    tokenizer: CustomTokenizer,
    decoding_method: str = 'greedy',
    beam_size = 5,
    max_len = 50,
    top_k = 5,
    top_p = 0.4,
    temperature = 0.7
):
    model.eval()
    src = tokenizer.text_to_tensor(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    if decoding_method == 'greedy':
        tgt_tokens = greedy_decode(
            model,
            src,
            src_mask,
            max_len,
            start_symbol=tokenizer.BOS_IDX,
            end_symbol=tokenizer.EOS_IDX
        )

    elif decoding_method == 'beam_search':
        tgt_tokens = beam_search_decode(
            model,
            src,
            src_mask,
            max_len,
            start_symbol=tokenizer.BOS_IDX,
            end_symbol=tokenizer.EOS_IDX,
            beam_size=beam_size
        )
    elif decoding_method == 'top_k':
        tgt_tokens = advanced_sampling_decode(
            model, src, src_mask, max_len, tokenizer, sampling_method=top_k_sampling, k=top_k
        )

    elif decoding_method == 'top_p':
        tgt_tokens = advanced_sampling_decode(
            model, src, src_mask, max_len, tokenizer, sampling_method=top_p_sampling, p=top_p
        )

    elif decoding_method == 'random_sampling':
        tgt_tokens = advanced_sampling_decode(
            model, src, src_mask, max_len, tokenizer, sampling_method=random_sampling, temperature=temperature
        )

    else:
        raise ValueError("Invalid decoding method. Choose 'greedy', 'beam_search', 'top_k', 'top_p', or 'random_sampling'.")

    tgt_tokens = tgt_tokens.flatten()
    outputted_text = tokenizer.tensor_to_text(tgt_tokens)
    return outputted_text.replace("<bos>", "").replace("<eos>", "")
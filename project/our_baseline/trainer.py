import os

import torch
from sklearn.metrics import accuracy_score
from torch import nn
import numpy as np
import logging
import torch
from tqdm import tqdm
import sys
sys.path.append(".")
from conll2002_metrics import conll2002_measure
from consts import PRINT_EVERY, LABEL_PAD_INDEX, slot_set

logger = logging.getLogger()

index2slot = ['O', 'B-weather/noun', 'I-weather/noun', 'B-location', 'I-location', 'B-datetime', 'I-datetime',
              'B-weather/attribute', 'I-weather/attribute', 'B-reminder/todo', 'I-reminder/todo',
              'B-alarm/alarm_modifier', 'B-reminder/noun', 'B-reminder/recurring_period', 'I-reminder/recurring_period',
              'B-reminder/reference', 'I-reminder/noun', 'B-reminder/reminder_modifier', 'I-reminder/reference',
              'I-reminder/reminder_modifier', 'B-weather/temperatureUnit', 'I-alarm/alarm_modifier',
              'B-alarm/recurring_period', 'I-alarm/recurring_period']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_dataloader, model, optimizer, tokenizer = None):
    print("---"*10+"Begin Train"+"---"*10)
    model.train()
    intent_criterion = nn.CrossEntropyLoss()
    slot_criterion = nn.CrossEntropyLoss(ignore_index=LABEL_PAD_INDEX)
    total_loss = 0
    intent_num_corr = 0
    intent_total = 0
    all_slot_conlls = []
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        if batch_idx % PRINT_EVERY == 0 and intent_total > 0:
            print(
                f'Done with {batch_idx}/{len(train_dataloader)} Loss {total_loss / intent_total} Intent acc: {intent_num_corr / intent_total} Slot F1: {conll2002_measure(all_slot_conlls)["fb1"]}')

        all_intent_labels, all_slot_label_mask, intent_preds, slot_preds = make_preds(batch, model)
        intent_loss = intent_criterion(intent_preds, all_intent_labels)

        #  all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask = [bS x seqLen]
        #  all_intent_label = [bS]
        #  intent_preds = [bS x seqLen]
        # slot_preds = [bs x seqLen x 24??]

        slot_loss = slot_criterion(slot_preds.contiguous().view(-1, slot_preds.shape[-1]), all_slot_label_mask.view(-1))
        loss = 0.7 * intent_loss + 0.3 * slot_loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        intent_total += all_intent_labels.shape[0]
        intent_num_corr += (torch.argmax(intent_preds, 1) == all_intent_labels).sum().item()
        text = tokenizer.batch_decode(batch[0].tolist()) if tokenizer is not None else None
        batch_slot_conlls = get_conll_prediction_from_model_predictions(all_slot_label_mask, slot_preds, text)
        all_slot_conlls.extend(batch_slot_conlls)
    print("---"*10+"End Train"+"---"*10)

def validate(valid_dataloader, model, tokenizer = None):
    print("---"*10+"Begin Validation"+"---"*10)
    model.eval()
    with torch.no_grad():
        intent_num_corr = 0
        intent_total = 0
        all_slot_conlls = []
        for batch_idx, batch in enumerate(valid_dataloader):
            if batch_idx % PRINT_EVERY == 0 and intent_total > 0:
                print(
                    f'Done with {batch_idx}/{len(valid_dataloader)} Intent acc: {intent_num_corr / intent_total} Slot F1: {conll2002_measure(all_slot_conlls)["fb1"]}')
            all_intent_labels, all_slot_label_mask, intent_preds, slot_preds = make_preds(batch, model)
            intent_total += all_intent_labels.shape[0]
            intent_num_corr += (torch.argmax(intent_preds, 1) == all_intent_labels).sum().item()
            # TODO: Figure out slot acc
            text = [tokenizer.convert_ids_to_tokens(x) for x in batch[0].tolist()] if tokenizer is not None else None
            batch_slot_conlls = get_conll_prediction_from_model_predictions(all_slot_label_mask, slot_preds, text)
            all_slot_conlls.extend(batch_slot_conlls)

    conllscore = conll2002_measure(all_slot_conlls)['fb1']
    print(f'Overall => Intent acc: {intent_num_corr / intent_total} Slot F1: {conllscore}')
    print("---"*10+"End Validation"+"---"*10)
    return {'intent_acc': intent_num_corr / intent_total, 'slot_f1': conllscore, 'output': all_slot_conlls}


def make_preds(batch, model):
    all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask, all_intent_labels = batch
    all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask, all_intent_labels = all_input_ids.to(
        device), all_attention_mask.to(device), all_token_type_ids.to(device), all_slot_label_mask.to(
        device), all_intent_labels.to(device)
    intent_preds, slot_preds = model.forward(all_input_ids, all_attention_mask, all_token_type_ids)
    return all_intent_labels, all_slot_label_mask, intent_preds, slot_preds


def get_conll_prediction_from_model_predictions(all_slot_label_mask,
                                                slot_preds, text = None):
    batch_size = all_slot_label_mask.shape[0]
    max_seq_len = all_slot_label_mask.shape[1]
    all_slot_conlls = []
    out_slot_label_list = [[] for _ in range(batch_size)]
    slot_preds_list = [[] for _ in range(batch_size)]
    slot_preds = torch.argmax(slot_preds, dim=2)
    slot_preds = slot_preds.detach()
    for i in range(batch_size):
        for j in range(max_seq_len):
            if all_slot_label_mask[i, j] != LABEL_PAD_INDEX:
                gold_slot_value = slot_set[all_slot_label_mask[i][j]]
                out_slot_label_list[i].append(gold_slot_value)
                pred_slot_value = slot_set[slot_preds[i][j]]
                slot_preds_list[i].append(pred_slot_value)
                word = text[i][j] if text is not None else 'w'
                all_slot_conlls.append(word + " " + pred_slot_value + " " + gold_slot_value)
        all_slot_conlls.append("\n")
    return all_slot_conlls

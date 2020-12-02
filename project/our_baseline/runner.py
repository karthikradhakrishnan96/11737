import argparse
import os

import torch
from torch import optim

import sys

from transformers import BertTokenizer

sys.path.append(".")
from consts import intent_set, slot_set
from dataloader_utils import get_nlu_dataloader
from models import BertBasedTODModel
from trainer import train, validate


def get_cmd_args():
    parser = argparse.ArgumentParser(description="Indic-TOD")
    parser.add_argument("--bS", type=int, default=32, help="Batch size")
    parser.add_argument("--clean", default=False, action="store_true", help="Clean text if store true")
    parser.add_argument("--train_langs", nargs='+', type=str, default="", help="")
    parser.add_argument("--test_lang", type=str, default="", help="")
    parser.add_argument("--bert_type", type=str, default='bert-base-uncased', help='')
    parser.add_argument("--max_seq_len", type=int, default=250, help='')
    parser.add_argument("--n_epoch", type=int, default=10, help='')
    parser.add_argument("--model_save_path", type=str, default='../output_dir', help='')
    parser.add_argument('--run_key', type=str, default='test')

    return parser.parse_args()


def get_model_and_opt(params):
    # TODO: Add pretraind stuff
    model = BertBasedTODModel(params.bert_type, len(intent_set), len(slot_set))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1.5e-5)
    return model, optimizer


if __name__ == "__main__":
    params = get_cmd_args()

    save_dir = params.model_save_path + "/" + params.run_key
    os.system(f'mkdir -p {save_dir}')
    tokenizer = BertTokenizer.from_pretrained(params.bert_type)
    train_loader, val_loader, test_loader = get_nlu_dataloader(params, tokenizer)
    model, optimizer = get_model_and_opt(params)
    best_intent_acc = -1
    best_slot_f1 = -1
    best_epoch = -1
    for epoch in range(1, params.n_epoch + 1):
        print(
            f'Training Epoch : {epoch}, best results so far  : {best_intent_acc}, {best_slot_f1} @ epoch  : {best_epoch} (by intent)')
        train(train_loader, model, optimizer, tokenizer)
        validation_results = validate(val_loader, model, tokenizer)
        if validation_results['intent_acc'] > best_intent_acc:
            best_epoch = epoch
            best_intent_acc = validation_results['intent_acc']
        if validation_results['slot_f1'] > best_slot_f1:
            best_slot_f1 = validation_results['slot_f1']

        if epoch == best_epoch:
            print('Saving model and opt')
            torch.save(model.state_dict(), save_dir + "/model_" + str(epoch) + ".pt")
            torch.save(optimizer.state_dict(), save_dir + "/opt_" + str(epoch) + ".pt")
            with open(save_dir + '/output_slot_outs_' + str(epoch) + '.conll', 'w') as f:
                f.write('\n'.join(validation_results['output']))

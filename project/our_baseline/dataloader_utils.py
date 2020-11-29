import csv
import json
import logging
import os
import re
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

from group.project.our_baseline.consts import LABEL_PAD_INDEX, intent_set, slot_set

logger = logging.getLogger()


# for dialogue NLU dataset
def preprocess_nlu_data(data, lang, clean_txt=True, token_mapping=None, vocab_path=None, filtered=False,
                        filtered_scale=None):
    # preprocess from raw (lang) data
    # print("============ Preprocess %s data ============" % lang)
    logger.info("============ Preprocess %s data ============" % lang)
    global intent_set, slot_set
    data_folder = os.path.join('./data/', lang)
    train_path = os.path.join(data_folder, "train-%s.tsv" % lang)
    eval_path = os.path.join(data_folder, "eval-%s.tsv" % lang)
    # test_path = os.path.join(data_folder, "test-%s.tsv" % lang)
    if lang != "en" and filtered == True:
        print("testing filtering data")
        test_path = os.path.join(data_folder, "test-%s.filter.%s.tsv" % (lang, filtered_scale))
    else:
        test_path = os.path.join(data_folder, "test-%s.tsv" % lang)

    data_train, _, _ = parse_tsv(train_path)
    data_eval, intent_set, slot_set = parse_tsv(eval_path, intent_set=intent_set, slot_set=slot_set, istrain=False)
    data_test, intent_set, slot_set = parse_tsv(test_path, intent_set=intent_set, slot_set=slot_set, istrain=False)

    assert len(intent_set) == len(set(intent_set))
    assert len(slot_set) == len(set(slot_set))

    # logger.info("number of intent in %s is %s" % (lang, len(intent_set)))
    # logger.info("number of slot in %s is %s" % (lang, len(slot_set)))
    # print("number of intent in %s is %s" % (lang, len(intent_set)))
    # print("number of slot in %s is %s" % (lang, len(slot_set)))

    if lang == "en" and token_mapping is not None:
        logger.info("generating mixed language training data")
        # data_train = gen_mix_lang_data(data_train, token_mapping)
        # data_eval = gen_mix_lang_data(data_eval, token_mapping)
        # data_eval = gen_mix_lang_data(data_eval, token_mapping)

    if clean_txt == True:
        # clean_data
        logger.info("cleaning data on %s language" % lang)
        data_train = clean_text(data_train, lang)
        data_eval = clean_text(data_eval, lang)
        data_test = clean_text(data_test, lang)

    # assert vocab_path is not None
    # logger.info("Loading vocab from %s" % vocab_path)
    # with open(vocab_path, "rb") as f:
    #     vocab = pickle.load(f)
    # logger.info("vocab size of %s is %d" % (lang, vocab.word_num))
    # print("vocab size of %s is %d" % (lang, vocab.word_num))
    vocab = None

    data_train_bin = binarize_nlu_data(data_train, intent_set, slot_set)
    data_eval_bin = binarize_nlu_data(data_eval, intent_set, slot_set)
    data_test_bin = binarize_nlu_data(data_test, intent_set, slot_set)
    data[lang] = {"train": data_train_bin, "eval": data_eval_bin, "test": data_test_bin, "vocab": vocab}


# for dialogue NLU dataset
def parse_tsv(data_path, intent_set=None, slot_set=None, istrain=True):
    """
    Input:
        data_path: the path of data
        intent_set: set of intent (empty if it is train data)
        slot_set: set of slot type (empty if it is train data)
    Output:
        data_tsv: {"text": [[token1, token2, ...], ...], "slot": [[slot_type1, slot_type2, ...], ...], "intent": [intent_type, ...]}
        intent_set: set of intent
        slot_set: set of slot type
    """
    if slot_set is None:
        slot_set = ["O"]
    if intent_set is None:
        intent_set = []
    slot_type_list = ["alarm", "datetime", "location", "reminder", "weather"]
    data_tsv = {"text": [], "slot": [], "intent": []}
    with open(data_path) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for i, line in enumerate(reader):
            intent = line[0]
            if istrain == True and intent not in intent_set: intent_set.append(intent)
            if istrain == False and intent not in intent_set:
                intent_set.append(intent)
                # logger.info("Found intent %s not in train data" % intent)
                # print("Found intent %s not in train data" % intent)
            slot_splits = line[1].split(",")
            slot_line = []
            slot_flag = True
            if line[1] != '':
                for item in slot_splits:
                    item_splits = item.split(":")
                    assert len(item_splits) == 3
                    # slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[2].split("/")[0]}
                    slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[2]}
                    flag = False
                    for slot_type in slot_type_list:
                        if slot_type in slot_item["slot"]:
                            flag = True

                    if flag == False:
                        slot_flag = False
                        break
                    # if istrain == True and slot_item["slot"] not in slot_set: slot_set.append(slot_item["slot"])
                    # if istrain == False and slot_item["slot"] not in slot_set:
                    #     slot_set.append(slot_item["slot"])
                    #     # logger.info("Found slot %s not in train data" % item_splits[2])
                    #     # print("Found slot %s not in train data" % item_splits[2])
                    slot_line.append(slot_item)

            if slot_flag == False:
                # slot flag not correct
                continue

            token_part = json.loads(line[4])
            tokens = token_part["tokenizations"][0]["tokens"]
            tokenSpans = token_part["tokenizations"][0]["tokenSpans"]

            data_tsv["text"].append(tokens)
            data_tsv["intent"].append(intent)
            slots = []
            for tokenspan in tokenSpans:
                nolabel = True
                for slot_item in slot_line:
                    start =  tokenspan["start"]
                    # if int(start) >= int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                    if int(start) == int(slot_item["start"]):
                        nolabel = False
                        slot_ = "B-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                    if int(start) > int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                        nolabel = False
                        slot_ = "I-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                if nolabel == True: slots.append("O")
            data_tsv["slot"].append(slots)

            assert len(slots) == len(tokens)

    return data_tsv, intent_set, slot_set


# for dialogue NLU dataset
def clean_text(data, lang):
    # detect pattern
    # detect <TIME>
    pattern_time1 = re.compile(r"[0-9]+[ap]")
    pattern_time2 = re.compile(r"[0-9]+[;.h][0-9]+")
    pattern_time3 = re.compile(r"[ap][.][am]")
    pattern_time4 = range(2000, 2020)
    # pattern_time5: token.isdigit() and len(token) == 3

    pattern_time_th1 = re.compile(r"[\u0E00-\u0E7F]+[0-9]+")
    pattern_time_th2 = re.compile(r"[0-9]+[.]*[0-9]*[\u0E00-\u0E7F]+")
    pattern_time_th3 = re.compile(r"[0-9]+[.][0-9]+")

    # detect <LAST>
    pattern_last1 = re.compile(r"[0-9]+min")
    pattern_last2 = re.compile(r"[0-9]+h")
    pattern_last3 = re.compile(r"[0-9]+sec")

    # detect <DATE>
    pattern_date1 = re.compile(r"[0-9]+st")
    pattern_date2 = re.compile(r"[0-9]+nd")
    pattern_date3 = re.compile(r"[0-9]+rd")
    pattern_date4 = re.compile(r"[0-9]+th")

    # detect <LOCATION>: token.isdigit() and len(token) == 5

    # detect <NUMBER>: token.isdigit()

    # for English: replace contain n't with not
    # for English: remove 's, 'll, 've, 'd, 'm
    remove_list = ["'s", "'ll", "'ve", "'d", "'m"]

    data_clean = {"text": [], "slot": [], "intent": []}
    data_clean["slot"] = data["slot"]
    data_clean["intent"] = data["intent"]
    for token_list in data["text"]:
        token_list_clean = []
        for token in token_list:
            new_token = token
            # detect <TIME>
            if lang != "th" and (bool(re.match(pattern_time1, token)) or bool(re.match(pattern_time2, token)) or bool(
                    re.match(pattern_time3, token)) or token in pattern_time4 or (token.isdigit() and len(token) == 3)):
                new_token = "<TIME>"
                token_list_clean.append(new_token)
                continue
            if lang == "th" and (
                    bool(re.match(pattern_time_th1, token)) or bool(re.match(pattern_time_th2, token)) or bool(
                re.match(pattern_time_th3, token))):
                new_token = "<TIME>"
                token_list_clean.append(new_token)
                continue
            # detect <LAST>
            if lang == "en" and (bool(re.match(pattern_last1, token)) or bool(re.match(pattern_last2, token)) or bool(
                    re.match(pattern_last3, token))):
                new_token = "<LAST>"
                token_list_clean.append(new_token)
                continue
            # detect <DATE>
            if lang == "en" and (bool(re.match(pattern_date1, token)) or bool(re.match(pattern_date2, token)) or bool(
                    re.match(pattern_date3, token)) or bool(re.match(pattern_date4, token))):
                new_token = "<DATE>"
                token_list_clean.append(new_token)
                continue
            # detect <LOCATION>
            if lang != "th" and (token.isdigit() and len(token) == 5):
                new_token = "<LOCATION>"
                token_list_clean.append(new_token)
                continue
            # detect <NUMBER>
            if token.isdigit():
                new_token = "<NUMBER>"
                token_list_clean.append(new_token)
                continue
            if lang == "en" and ("n't" in token):
                new_token = "not"
                token_list_clean.append(new_token)
                continue
            if lang == "en":
                for item in remove_list:
                    if item in token:
                        new_token = token.replace(item, "")
                        break

            token_list_clean.append(new_token)

        assert len(token_list_clean) == len(token_list)
        data_clean["text"].append(token_list_clean)

    return data_clean





def binarize_nlu_data(data, intent_set, slot_set):
    data_bin = {"text": [], "slot": [], "intent": []}
    # binarize intent
    for intent in data["intent"]:
        index = intent_set.index(intent)
        data_bin["intent"].append(index)
    # binarize text
    for text_tokens in data["text"]:
        text_bin = []
        for token in text_tokens:
            text_bin.append(token)
        data_bin["text"].append(text_bin)
    # binarize slot
    for slot in data["slot"]:
        slot_bin = []
        for slot_item in slot:
            index = slot_set.index(slot_item)
            slot_bin.append(index)
        data_bin["slot"].append(slot_bin)

    assert len(data_bin["slot"]) == len(data_bin["text"]) == len(data_bin["intent"])
    for text, slot in zip(data_bin["text"], data_bin["slot"]):
        assert len(text) == len(slot)

    return data_bin


class Dataset(data.Dataset):
    def __init__(self, data):
        self.X = data["text"]
        self.y1 = data["intent"]
        self.y2 = data["slot"]

    def __getitem__(self, index):
        return self.X[index], self.y1[index], self.y2[index]

    def __len__(self):
        return len(self.X)


class Dataset2(data.Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return tuple([value[index] for value in self.values])

    def __len__(self):
        return len(self.values[0])



def collate_fn2(data, tokenizer):
    all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask, all_intent_labels = zip(*data)
    lengths = [len(input_ids) for input_ids in all_input_ids]
    max_length_batch = max(lengths)
    all_input_ids_padded = []
    all_attention_mask_padded = []
    all_token_type_ids_padded = []
    all_slot_label_mask_padded = []
    for i in range(len(all_input_ids)):
        all_input_ids_padded.append(all_input_ids[i] + [tokenizer.pad_token_id]*(max_length_batch - len(all_input_ids[i])))
        all_attention_mask_padded.append(all_attention_mask[i] + [tokenizer.pad_token_id]*(max_length_batch - len(all_input_ids[i])))
        all_token_type_ids_padded.append(all_token_type_ids[i] + [tokenizer.pad_token_id]*(max_length_batch - len(all_input_ids[i])))
        all_slot_label_mask_padded.append(all_slot_label_mask[i] + [LABEL_PAD_INDEX]*(max_length_batch - len(all_input_ids[i])))
    return torch.LongTensor(all_input_ids_padded), torch.LongTensor(all_attention_mask_padded), torch.LongTensor(all_token_type_ids_padded), torch.LongTensor(all_slot_label_mask_padded), torch.LongTensor(all_intent_labels)


def load_data(params):
    data = {lang: {} for lang in params.train_langs}
    data[params.test_lang] = {}
    data = {"en": {}, "es": {}, "th": {}}
    for train_lang in params.train_langs:
        preprocess_nlu_data(data, train_lang, params.clean)
    # load Transfer language data
    preprocess_nlu_data(data, params.test_lang, params.clean)

    return data


def make_bert_compatible_data(data, tokenizer, params):
    # data : X, y1 (intent), y2 (slot labels)
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []
    all_intent_labels = []
    for words, intent_label, slot_labels in zip(data['text'], data['intent'], data['slot']):
        tokens = []
        slot_label_mask = []
        for word, slot_label in zip(words, slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([slot_label] + [LABEL_PAD_INDEX] * (len(word_tokens) - 1))
        special_tokens_count = 2
        if len(tokens) > params.max_seq_len - special_tokens_count:
            tokens = tokens[: (params.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(params.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [0] * len(tokens)
        slot_label_mask += [LABEL_PAD_INDEX]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [0] + token_type_ids
        slot_label_mask = [LABEL_PAD_INDEX] + slot_label_mask
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)
        all_intent_labels.append(intent_label)

    return all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask, all_intent_labels


def get_nlu_dataloader(params):
    data = load_data(params)
    tokenizer = BertTokenizer.from_pretrained(params.bert_type)
    train_data = data[params.train_langs[0]]["train"]
    val_data = data[params.test_lang]["eval"]
    test_data = data[params.test_lang]["test"]


    train_dataset = Dataset2(make_bert_compatible_data(train_data, tokenizer, params))
    val_dataset = Dataset2(make_bert_compatible_data(val_data, tokenizer, params))
    test_dataset = Dataset2(make_bert_compatible_data(test_data, tokenizer, params))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=params.bS, shuffle=True, collate_fn=lambda x : collate_fn2(x, tokenizer))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=params.bS, shuffle=False, collate_fn=lambda x : collate_fn2(x, tokenizer))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=params.bS, shuffle=False, collate_fn=lambda x : collate_fn2(x, tokenizer))
    return train_dataloader, val_dataloader, test_dataloader



    # # TODO: Only one for now. Maybe extend later
    # dataset_tr = Dataset(data[params.train_langs[0]]["train"])
    # dataset_val = Dataset(data[params.test_lang]["eval"])
    # dataset_test = Dataset(data[params.test_lang]["test"])
    #
    # dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=params.bS, shuffle=True)
    # dataloader_val = DataLoader(dataset=dataset_val, batch_size=params.bS, shuffle=False)
    # dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.bS, shuffle=False)
    # return dataloader_tr, dataloader_val, dataloader_test

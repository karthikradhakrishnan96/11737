import json
import numpy as np
from tqdm import tqdm
import editdistance
import os
from transformers import BertModel, BertTokenizer
import torch

model = BertModel.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def get_longest_continuous_sequence(sequence):
    return max(np.split(sequence, np.where(np.diff(sequence) != 1)[0]+1), key=len).tolist()

# Add the aligned file from MNLP alignment ipynb here if word align method
# with open('/Users/kai/PycharmProjects/11737/group/project/alignment/aligned_test.txt') as f:
#     aligns = f.read().split('\n')


BERT_THRESHOLD = 0
def get_word_probs2(s1, s2, wi):
    sub_tokens = [tokenizer.cls_token]
    words = s1.split(' ')
    word_sub_index = []
    start = 1 # cls is 0
    for p1 in words:
        subwords = tokenizer.tokenize(p1)
        word_sub_index.append(start)
        start += len(subwords)
        sub_tokens.extend(subwords)




    input_ids = [tokenizer._convert_token_to_id(subtoken) for subtoken in sub_tokens]
    # input_ids = [tokenizer.cls_token_id] + input_ids
    sequence_output, cls1 = model(input_ids=torch.LongTensor([input_ids]), attention_mask=None,
                                  token_type_ids=None)

    sequence_output = sequence_output[0]

    word_encodings = []
    for i in range(len(word_sub_index)):
        start = word_sub_index[i]
        end = word_sub_index[i+1] if i+1 < len(word_sub_index) else len(sequence_output)
        end = start + 1
        word_encodings.append(torch.mean(sequence_output[start:end], dim=0))

    # word_encodings = sequence_output[word_sub_index]

    sub_tokens = []
    words = s2.split(' ')
    word_sub_index = []
    start = 1  # cls is 0
    interest_end = -1
    interest = None
    for word_index_curr, p1 in enumerate(words):
        subwords = tokenizer.tokenize(p1)
        word_sub_index.append(start)
        if interest:
            interest_end = start
        if word_index_curr == wi:
            interest = start
        start += len(subwords)
        sub_tokens.extend(subwords)

    input_ids = [tokenizer._convert_token_to_id(subtoken) for subtoken in sub_tokens]
    input_ids = [tokenizer.cls_token_id] + input_ids
    sequence_output, cls2 = model(input_ids=torch.LongTensor([input_ids]), attention_mask=None,
                                  token_type_ids=None)
    interest_end = interest + 1
    token_encoding = torch.mean(sequence_output[0][interest:interest_end], dim=0)
    sims = []
    for encoding in word_encodings:
        sims.append(torch.cosine_similarity(encoding, token_encoding, dim = -1).item())
    return sims





def get_matching_token_location(token, token_list, mode="edit_distance", all_phrase_tokens = None, wi = 0):
    # This logic can change (Direct Match, Edit Distance based match etc)

    if mode=="direct":
        try:
            return token_list.index(token)
        except:
            return -1
    elif mode=="edit_distance":
        edit_distances = [editdistance.eval(token, sent_token) for sent_token in token_list]
        max_prefix_lengths = [len(os.path.commonprefix([token, sent_token])) for sent_token in token_list]
        if max(max_prefix_lengths) > 2: #TODO : Vary this threshold
            return max_prefix_lengths.index(max(max_prefix_lengths))
        elif min(edit_distances) == 0: #TODO : Vary this threshold (Maybe < 2 or smtn)
            return edit_distances.index(min(edit_distances))
        else:

            # return -1 # Uncomment to not use mBERT
            sims = get_word_probs2(' '.join(token_list), all_phrase_tokens, wi)
            return np.argmax(sims) if np.max(sims) > BERT_THRESHOLD else -1

# 136
def get_valid_token_list(candidate_tokens, mode="continuous"):
    # This logic can change (Continuous spans; Expanding Non-Continuous spans)

    if mode=="continuous":
        return get_longest_continuous_sequence(sorted(candidate_tokens))
    elif mode=="expand":
        ordered_tokens = sorted(candidate_tokens)
        if candidate_tokens:
            return [token_index for token_index in range(ordered_tokens[0], ordered_tokens[-1]+1)]
        else:
            return []


file_path = "/Users/kai/PycharmProjects/11737/group/project/our_dataprep/train-es-pre20.json"
# output_file_path = "D:\CMU\Courses\\11737\Project\dataset\\ta\\eval-annotated-ta.json"
output_file_path = "/Users/kai/PycharmProjects/11737/group/project/our_dataprep/outs/train-es-align-again-2-thresh20.json"

token_matching_mode = "edit_distance" #"direct"
token_span_mode = "expand" #"continuous"

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for index, datapoint in tqdm(enumerate(data)):
    lrl_sent = datapoint['lrl_transl']
    lrl_sent_tokens = lrl_sent.split(' ')
    # TODO: need to fix cases like 8: 45



    lrl_token_labels = {}
    # Uncomment these if word alignment is to be used
    # alignment = aligns[index]
    # alignment = {int(x) : int(y) for x,y in [z.split('-') for z in alignment.split(' ')]}

    for slot_info in datapoint['annotation_phrases']:
        matching_token_positions = []
        # en_start, en_end = slot_info['en_start'], slot_info['en_end']
        # en_words = datapoint['en_sent'].split(' ')
        # curr_en = 0
        # curr_w = 0
        # try:
        #     while curr_en != en_start:
        #         curr_en += len(en_words[curr_w])
        #         curr_en += 1
        #         curr_w += 1
        #     start_w = curr_w
        #     orig_curr_en = curr_en
        #     while curr_en + len(en_words[curr_w]) < en_end:
        #         curr_en += len(en_words[curr_w])
        #         curr_en += 1
        #         curr_w += 1
        # except:
        #     print('panik')
        #     print(' '.join(en_words), slot_info['en_phrase'])
        #     continue
        # end_w = curr_w
        # lrl_tokens = [lrl_sent_tokens[alignment[x]] for x in range(start_w, end_w+1) if x in alignment]
        lrl_tokens = slot_info['lrl_transl'].split(' ')
        for word_index, word in enumerate(lrl_tokens):
            matching_token_index = get_matching_token_location(word, lrl_sent_tokens, mode=token_matching_mode, all_phrase_tokens = ' '.join(lrl_tokens), wi = word_index)
            if matching_token_index > -1:
                matching_token_positions.append(matching_token_index)

        # Identifying valid tokens
        valid_token_indices = get_valid_token_list(matching_token_positions, mode=token_span_mode)

        # Identifying and updating character level span start and end
        slot_tokens = list(map(lrl_sent_tokens.__getitem__, valid_token_indices))
        slot_span = ' '.join(slot_tokens)

        slot_span_start = -1
        slot_span_end = -1

        if len(slot_span) > 0:
            try:
                slot_span_start = lrl_sent.index(slot_span)
                slot_span_end = slot_span_start + len(slot_span)
            except:
                pass

            # Tag labels for each word

            lrl_token_labels[slot_tokens[0]] = "B-" + slot_info['slot_type']
            for slot_token in slot_tokens[1:]:
                lrl_token_labels[slot_token] = "I-" + slot_info['slot_type']

        slot_info['lrl_start'] = slot_span_start
        slot_info['lrl_end'] = slot_span_end


    # Adding the labels to all the LRL Tokens
    datapoint['lrl_tokens'] = []
    for lrl_sent_token in lrl_sent_tokens:
        lrl_token_start = lrl_sent.index(lrl_sent_token)
        lrl_token_end = lrl_token_start + len(lrl_sent_token)
        lrl_token_label = lrl_token_labels.get(lrl_sent_token, "NoLabel")

        datapoint['lrl_tokens'].append({lrl_sent_token : {'start' : lrl_token_start, 'end' : lrl_token_end, 'slot_type' : lrl_token_label}})


with open(output_file_path, 'w', encoding='utf-8') as g:
    json.dump(data, g, ensure_ascii=False)


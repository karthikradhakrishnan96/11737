import json
import csv
import shortuuid as uuid
from g_translate import translate_text
import pickle
from tqdm import tqdm

def batchify(data_list, batch_size):
    return [data_list[pos : pos + batch_size] for pos in range(0, len(data_list), batch_size)]

eval_data_path = "/Users/kai/PycharmProjects/11737/group/project/mlt_baseline/mixed-language-training/data/nlu/nlu_data/en/train-en.tsv" # test_file_path #"D:\CMU\Courses\\11737\Project\dataset\en\eval-en.tsv"
eval_conll_data_path = "/Users/kai/PycharmProjects/11737/group/project/mlt_baseline/mixed-language-training/data/nlu/nlu_data/en/train-en.conllu" #test_conll_file_path #"D:\CMU\Courses\\11737\Project\dataset\en\eval-en.conllu"
eval_ta_file_path = "/Users/kai/PycharmProjects/11737/group/project/our_dataprep/outs/train-ta-ourmt.pkl" #test_ta_file_path #"D:\CMU\Courses\\11737\Project\dataset\kn\\eval-kn.pkl"
eval_ta_file_path_json = "/Users/kai/PycharmProjects/11737/group/project/our_dataprep/outs/train-ta-ourmt.json" #test_ta_file_path_json #"D:\CMU\Courses\\11737\Project\dataset\kn\\eval-kn.json"
eval_ta_file_path_txt = "/Users/kai/PycharmProjects/11737/group/project/our_dataprep/outs/train-ta-ourmt.txt" #test_ta_file_path_txt #"D:\CMU\Courses\\11737\Project\dataset\kn\\eval-kn.txt"


eval_connll_data = []
eval_data = []
final_data = []

en_sent_data = []
en_annotation_data = []
example_limit = 50000

utterance_set = set()

with open(eval_conll_data_path, 'r') as f:
    raw_data = f.read()
    eval_connll_data = raw_data.split('\n\n')

with open(eval_data_path, 'r') as g:
    raw_data = csv.reader(g, delimiter='\t')
    for row in raw_data:
        eval_data.append(row)

for info, conll_info in tqdm(zip(eval_data[:example_limit], eval_connll_data[:example_limit])):

    if info[2] in utterance_set:
        continue
    else:
        utterance_set.add(info[2])

    example = {}
    conll_info = conll_info.split('\n')

    example['en_sent'] = info[2]
    example['lrl_transl'] = "Translated English sentence into LRL"
    example['intent_class'] = info[0]

    example['en_tokens'] = []
    for word_info_list, span_info_list in zip(conll_info[3:], json.loads(info[4])['tokenizations'][0]['tokenSpans']):
        token_data = {}
        _, word, intent_type, slot_type = word_info_list.split('\t')
        span_start = span_info_list['start']
        span_end = span_start + span_info_list['length']
        token_data[word] = {'start':span_start, 'end':span_end, 'intent_type':intent_type, 'slot_type':slot_type}

        example['en_tokens'].append(token_data)

    example['annotation_phrases'] = []
    for slot_annotation in info[1].split(','):
        if len(slot_annotation) > 0:
            span_start, span_end, slot_type = slot_annotation.split(':')
            span_start, span_end = int(span_start), int(span_end)
            annotation_phrase = info[2][span_start:span_end]
            lrl_translation = "Translated English Annotation into LRL"

            annotation_uuid = uuid.uuid()
            example['annotation_phrases'].append({'en_start':span_start, 'en_end':span_end, 'en_phrase':annotation_phrase, 'lrl_start':-1, 'lrl_end' :-1, 'lrl_transl':lrl_translation, 'slot_type': slot_type, 'uuid':annotation_uuid})
            en_annotation_data.append((annotation_uuid, annotation_phrase))
        else:
            pass

    sent_uuid = uuid.uuid()
    example['uuid'] = sent_uuid
    en_sent_data.append((sent_uuid, info[2]))
    final_data.append(example)


en_sent_to_translate = [sentence for _, sentence in en_sent_data]
en_sent_ids = [idx for idx, _ in en_sent_data]

en_phrase_to_translate = [phrase for _, phrase in en_annotation_data]
en_phrase_ids = [idx for idx, _ in en_annotation_data]


# save the sent, phrase and ids to translate offline with a MT system and then load the translations from mt

# with open('en_sents.txt', 'w') as f:
#     f.write('\n'.join(en_sent_to_translate))
#
#
# with open('en_sent_ids.txt', 'w') as f:
#     f.write('\n'.join(en_sent_ids))
#
#
# with open('en_phrases.txt', 'w') as f:
#     f.write('\n'.join(en_phrase_to_translate))
#
#
# with open('en_phrase_ids.txt', 'w') as f:
#     f.write('\n'.join(en_phrase_ids))


with open('en_sent_and_phrase_ids.txt') as f:
    sent_phrase_ids = f.read().split('\n')
    sent_phrase_ids = {i:idx.strip() for i,idx in enumerate(sent_phrase_ids)}


# load from MT
transl_ids_map = {}
with open('./outs/translations_from_our_mt.log') as f:
    lines = f.read().split("\n")
    for line in lines:
        if line.startswith('H'):
            line = line.split('\t', maxsplit=2)
            line_no, line_transl = line[0][2:], line[-1]
            line_idx = sent_phrase_ids[int(line_no)]
            transl_ids_map[line_idx] = line_transl

lrl_sent_data = lrl_phrase_data = transl_ids_map

#Slightly messed up
with open('en_sent_and_phrases.txt') as f:
    sent_phrases = f.read().split('\n')
    sent_phrases = {phrase:idx for idx,phrase in enumerate(sent_phrases)}

# To use google translate
# lrl_translated_list = []
# for sent_batch in batchify(en_sent_to_translate, 50):
#     lrl_translated_list.extend(translate_text(sent_batch, 'translation-296703'))
#
# lrl_sent_data = {uuid_token : lrl_sent for (uuid_token, en_sent), lrl_sent in zip(en_sent_data, lrl_translated_list)}
#
# en_phrase_to_translate = [phrase for _, phrase in en_annotation_data]
# lrl_translated_list = []
# for phrase_batch in batchify(en_phrase_to_translate, 50):
#     lrl_translated_list.extend(translate_text(phrase_batch, 'translation-296703'))
#
# lrl_phrase_data = {uuid_token : lrl_phrase for (uuid_token, en_phrase), lrl_phrase in zip(en_annotation_data, lrl_translated_list)}

for datapoint in final_data:
    datapoint['lrl_transl'] = lrl_sent_data[sent_phrase_ids[sent_phrases[datapoint['en_sent']]]]

    for annotation in datapoint['annotation_phrases']:
        annotation['lrl_transl'] = lrl_phrase_data[sent_phrase_ids[sent_phrases[annotation['en_phrase']]]] #lrl_phrase_data[annotation['uuid']]

with open(eval_ta_file_path, 'wb') as f:
    pickle.dump(final_data, f)

with open(eval_ta_file_path_json, 'w', encoding='utf-8') as g:
    json.dump(final_data, g, ensure_ascii=False)


with open(eval_ta_file_path_txt, 'w', encoding='utf-8') as h:
    for idx, datapoint in enumerate(final_data):
        h.write("Example Number : "+ str(idx) + "\n\n")
        h.write(datapoint['en_sent'] + " , " + datapoint['lrl_transl'] + "\n")
        for ann in datapoint['annotation_phrases']:
            h.write(ann['en_phrase'] + " , " + ann['lrl_transl'] + "\n")

        h.write("\n*********************************************************************\n\n")

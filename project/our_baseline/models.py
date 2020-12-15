import torch.nn as nn
from transformers import BertModel


class BertBasedTODModel(nn.Module):
    def __init__(self, bert_type, num_intent_labels, num_slot_labels):
        super(BertBasedTODModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_type)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.bert_output_dim = 768
        self.intent_classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.bert_output_dim, self.num_intent_labels))
        self.slot_classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.bert_output_dim, self.num_slot_labels))

    def forward(self, input_ids, attention_mask, token_type_ids):
        sequence_output, cls = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                                               token_type_ids=token_type_ids)

        intent_preds = self.intent_classifier(cls)
        slot_preds = self.slot_classifier(sequence_output)
        return intent_preds, slot_preds
import torch
from torch import nn
import transformers
from transformers.activations import ACT2FN
if transformers.__version__ == '3.4.0':
    from transformers.modeling_roberta import (
        RobertaEmbeddings,
        RobertaEncoder,
        RobertaPreTrainedModel,
        RobertaPooler,
    )
else:
    # the latest version
    from transformers.models.roberta.modeling_roberta import(
        RobertaEmbeddings,
        RobertaEncoder,
        RobertaPreTrainedModel,
        RobertaPooler,
    )

class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_attentions=None, output_hidden_states=None):
        assert input_ids is not None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output

class RobertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class RobertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = RobertaPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class RobertaPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RobertaLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, config.tar_dim)
    
    def forward(self, sequence_ouutput, pooled_output):
        prediction_scores = self.predictions(sequence_ouutput)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class RobertaForPreTraining(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.cls = RobertaPreTrainingHeads(config)
        self.config = config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        batch_size = input_ids.shape[0]
        sequence_output, pooled_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        pred_score = seq_relationship_score.argmax(dim=1)
        accuracy = torch.sum(pred_score.view(-1) == label.view(-1)).item()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        acc_part = dict()
        acc_part['tp'] = dict()
        acc_part['fp'] = dict()
        for i in range(self.config.tar_dim):
            acc_part['tp'][i] = 0
            acc_part['fp'][i] = 0
        for i in range(batch_size):
            score_i = pred_score[i].cpu()
            for j in range(self.config.tar_dim):
                if score_i == j:
                    acc_part['tp'][j] += list((pred_score.view(-1) == label.view(-1)))[i]
                    acc_part['fp'][j] += list((pred_score.view(-1) != label.view(-1)))[i]

        loss = loss_fct(seq_relationship_score.view(-1, self.config.tar_dim), label.view(-1))
        return loss, accuracy, acc_part

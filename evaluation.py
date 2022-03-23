"""The training loss goes down and we can see that the Acurracy on the test set also improves nicely. Because this notebook is just for demonstration purposes, we can stop here.

The resulting model of this notebook has been saved to [m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition](https://huggingface.co/m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition)

As a final check, let's load the model and verify that it indeed has learned to recognize the emotion in the speech.

Let's first load the pretrained checkpoint.

## Evaluation
"""

import librosa
from sklearn.metrics import classification_report
from transformers import AutoConfig, Wav2Vec2Processor
from datasets import load_dataset
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from nested_array_catcher import nested_array_catcher

# from emotion_recognition_wav2vec2 import Wav2Vec2ForSpeechClassification, Wav2Vec2ClassificationHead
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


test_dataset = load_dataset("csv", data_files={"test": "./content/data/test.csv"}, delimiter="\t")["test"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

processor_name_or_path = "m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition"
model_name_or_path = "./content/wav2vec2-xlsr-greek-speech-emotion-recognition/checkpoint-1210"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(processor_name_or_path)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)

    speech_array = nested_array_catcher(speech_array)

    # for i in range(len(speech_array)):
    #     try:
    #         assert isinstance(speech_array[i], np.float32)
    #     except Exception as e:
    #         print(e)
    #         print(294, type(speech_array[i])) # <class 'numpy.ndarray'>
    #         if isinstance(speech_array[i], np.ndarray):
    #             print(speech_array.shape, speech_array[i].shape, speech_array[i])
    #         speech_array = speech_array[i]
    #         print('new type: ', type(speech_array), type(speech_array[i]), speech_array, speech_array[i])

    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    # print(168, input_values.shape, len(input_values.shape)) # Tensor [batch_size=8, some_size], 8 1d tensors

    for i in range(len(input_values)):
        # input_values[i] = nested_array_catcher(input_values[i], target_type=np.float32)
        # print(186, input_values[i].shape, len(input_values[i].shape), input_values[i]) # 1d tensor
        if len(input_values[i].shape) > 1 :
            print(186, i, len(input_values[i].shape), input_values[i].shape, input_values[i])
        for j in range(len(input_values[i])):
            # print(188, len(input_values[i][j].shape)) #  type(input_values[i][j]), input_values[i][j],
            if len(input_values[i][j].shape) > 1 :
                print(188, i,j, len(input_values[i][j].shape), input_values[i][j].shape, input_values[i][j])
            # try:
            #     assert isinstance(input_values[i][j], np.float32)
            # except Exception as e:
            #     print(e)
            #     print(i, j) # 2791 0, 2791 1, 5097 0, 5097 1
            #     print(294, type(input_values[i]), input_values[i].shape, type(input_values[i][j]), input_values[i][j].shape) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
            #     if isinstance(input_values[i][j], np.ndarray):
            #         print(input_values[i].size, input_values[i][j].size, input_values[i][j])
            #     input_values[i] = input_values[i][j]
            #     print('new type: ', type(input_values[i][j]), input_values[i][j], input_values[i])

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

result = test_dataset.map(predict, batched=True, batch_size=8)

label_names = [config.id2label[i] for i in range(config.num_labels)]
label_names

y_true = [config.label2id[name] for name in result["emotion"]]
y_pred = result["predicted"]

print(y_true[:5])
print(y_pred[:5])

print(classification_report(y_true, y_pred, target_names=label_names))

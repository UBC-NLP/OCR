import pandas as pd
from evaluate import load
from datasets import load_dataset
from transformers import (
 EarlyStoppingCallback,
 default_data_collator,
 Seq2SeqTrainer, 
 Seq2SeqTrainingArguments,
 VisionEncoderDecoderModel,
 TrOCRProcessor,
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process

import torch
from torch.utils.data import Dataset
from PIL import Image

class IAMDataset(Dataset):
    def __init__(self, df, processor, max_target_length=128):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = "/home/elmadany/OCR/PATS01/Images/"+self.df['image'][idx]
        text = str(self.df['text'][idx])
        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model_name="microsoft/trocr-base-printed"
# from transformers import VisionEncoderDecoderModel

# initialize a vit-bert from a pretrained ViT and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
# model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
#     "google/vit-base-patch16-384", "UBC-NLP/MARBERTv2"
# )
# model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
# model.config.pad_token_id = processor.tokenizer.pad_token_id
# # make sure vocab size is set correctly
# model.config.vocab_size = model.config.decoder.vocab_size
# # saving model after fine-tuning
# model.save_pretrained("./vit-marbertv2")
# load fine-tuned model
model = VisionEncoderDecoderModel.from_pretrained(model_name)

df_train = pd.read_csv("/home/elmadany/OCR/PATS01/train.tsv", sep="\t")
df_dev = pd.read_csv("/home/elmadany/OCR/PATS01/valid.tsv", sep="\t")
df_test = pd.read_csv("/home/elmadany/OCR/PATS01/test.tsv", sep="\t")

train_dataset = IAMDataset(df=df_train, processor=processor)
eval_dataset = IAMDataset(df=df_dev,processor=processor)
test_dataset = IAMDataset(df=df_test,processor=processor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))
print("Number of test examples:", len(eval_dataset))

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4



training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    fp16=True, 
    # output_dir="./trocr_outputs",
    output_dir="trocr_output",
    load_best_model_at_end= True,
    metric_for_best_model="cer",
    greater_is_better= False,
    
)

cer_metric = load("cer")
wer_metric = load("wer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}

early_stopping_num=3

try:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
except:
    last_checkpoint=None

print ("last_checkpoint", last_checkpoint)
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    
)
trainer.add_callback(EarlyStoppingCallback(early_stopping_num)) #number of patient epochs before early stopping

trainer.train(resume_from_checkpoint=last_checkpoint)



test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
        )

metrics = test_results.metrics
print ("test", metrics)

test_preds = processor.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
test_preds = [pred.strip() for pred in test_preds]
print (test_preds)
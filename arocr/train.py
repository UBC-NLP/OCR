from datasets import load_metric, DatasetDict, load_dataset
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    AutoFeatureExtractor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
import argparse
import wandb

wandb.init(project="arocr", entity="gagan3012")

def preprocess(examples, processor, max_target_length=128):
    text = examples["text"]
        # prepare image (i.e. resize + normalize)
    image = examples["image"].convert("RGB")
        # image = Image.open(self.root_dir + file_name).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
    labels = processor.tokenizer(
            text, padding="max_length", max_length=max_target_length
    ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
    labels = [
            label if label != processor.tokenizer.pad_token_id else -100 for label in labels
    ]
    encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
    return encoding

def main(args):
    encoder = args.ENCODER
    decoder = args.DECODER
    model_name = args.MODEL_NAME
    DATASET = args.DATASET
    DATA_DIR = args.DATA_DIR

    print(encoder, decoder, model_name, DATA_DIR)

    dataset = load_dataset("/project/6007993/DataBank/OCR_data/Datasets/al/_Ready/AraOCR_dataset", DATASET,
                               cache_dir="./ocr_cache")

    # df = get_dataset(DATA_DIR, DATASET)
    # print(row_dataset)
    
    # test_df1 = load_dataset(DATA_DIR, split="test")

    tokenizer = AutoTokenizer.from_pretrained(decoder)

    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder)

    processor = TrOCRProcessor(feature_extractor, tokenizer)

    dataset.map(preprocess,
                processor=processor,
                batched=True)
    
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    eval_dataset = dataset["valid"]

    def model_init():
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)
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
        return model

    # set special tokens used for creating the decoder_input_ids from the labels

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=True,
        output_dir=model_name,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-08,
        save_steps=500,  # 4,  ###
        eval_steps=500,  # 4,  ###
        logging_steps=500,  # 4,  ###
        num_train_epochs=5,
        save_total_limit=1,
        weight_decay=0.005,
        learning_rate=3e-5,
        seed=42,
        report_to="wandb",
    )

    cer_metric = load_metric("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model_init=model_init,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    processor.save_pretrained(model_name)
    trainer.create_model_card(model_name=model_name)
    trainer.save_model(model_name)
    predict_results = trainer.predict(
            test_dataset, metric_key_prefix="cer"
        )
    metrics = predict_results.metrics
    print(metrics)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ENCODER",
        required=True,
        help=(
            "The encoder model to use. This can be a pretrained model name or a path to a model"
            " config file."
        ),
    )
    parser.add_argument(
        "--DECODER",
        required=True,
        help=(
            "The decoder model to use. This can be a pretrained model name or a path to a model"
            " config file."
        ),
    )
    parser.add_argument(
        "--DATASET",
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--MODEL_NAME",
        required=True,
        default=".",
        help="Path for saving training logs and predictions",
    )
    parser.add_argument(
        "--SEED",
        required=True,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--DATA_DIR",
        required=True,
        help="Path to the dataset",
    )
    args = parser.parse_args()

    # ------ CALL MAIN ------
    main(args)

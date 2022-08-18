from operator import contains
from datasets import load_metric, load_dataset, concatenate_datasets, interleave_datasets, DatasetDict
from transformers import (
    TrOCRProcessor,
    AutoFeatureExtractor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    HfArgumentParser,
    EarlyStoppingCallback
)
#import wandb
from dataclasses import dataclass, field
from typing import Optional
import os
from torch.utils.data import Dataset
from torch import tensor
from torch.cuda import is_available
import pandas as pd
from io import BytesIO
import base64
from IPython.core.display import HTML

#wandb.init(project="arocr", entity="gagan3012", settings=wandb.Settings(start_method="fork"))


def preprocess(examples, processor, max_target_length=512):
    text = examples["text"]
    image = examples["image"].convert("RGB")
    image = image.resize((224, 224))
    pixel_values = processor(image, return_tensors="pt").pixel_values
    labels = processor.tokenizer(
        text, padding="max_length", max_length=max_target_length
    ).input_ids
    labels = [label if label !=
              processor.tokenizer.pad_token_id else -100 for label in labels]
    encoding = {"pixel_values": pixel_values.squeeze(),
                "labels": tensor(labels)}
    return encoding


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    use_encoder_decoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use encoder-decoder model"}
    )
    encoder_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    decoder_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class TrainingArguments:
    per_device_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Adam."},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={
                      "help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."},
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    split: Optional[float] = field(
        default=1,
        metadata={
            "help": (
                "The data split"
            )
        },
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    encoder = model_args.encoder_model_name_or_path
    decoder = model_args.decoder_model_name_or_path
    model_name = model_args.model_name_or_path
    if encoder is None and decoder is None:
        encoder_split = "deit-base"
        decoder_split = "arbert-finetune"
    else:
        encoder_split = "encoder"
        decoder_split = "decoder"

    output_dir = "{}/{}-{}-{}".format(train_args.output_dir,
                                      data_args.dataset_config_name,
                                      decoder_split,
                                      encoder_split)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        per_device_eval_batch_size=train_args.per_device_eval_batch_size,
        fp16=is_available(),
        num_train_epochs=train_args.num_train_epochs,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        do_train=True,
        do_eval=True,
        do_predict=True,
        output_dir=output_dir,
        save_total_limit=2,
        run_name="{}-{}-{}".format(data_args.dataset_config_name,
                                   decoder_split,
                                   encoder_split)
    )

    print(model_args, data_args, training_args)

    adab = load_dataset(
        "/home/gagan30/scratch/arocr/AraOCR_dataset", "ADAB")#, cache_dir="/home/gagan30/scratch/arocr/cache")

    khatt = load_dataset(
        "/home/gagan30/scratch/arocr/AraOCR_dataset", "OnlineKhatt")#, cache_dir="/home/gagan30/scratch/arocr/cache")

    alexuw = load_dataset(
        "/home/gagan30/scratch/arocr/AraOCR_dataset", "alexuw")#, cache_dir="/home/gagan30/scratch/arocr/cache")

    merged_train = concatenate_datasets(
        [adab['train'], khatt['train'], alexuw['train']])
    merged_valid = interleave_datasets(
        [adab['validation'], khatt['validation'], alexuw['validation']], seed=42)
    merged_test = interleave_datasets(
        [adab['test'], khatt['test'], alexuw['test']], seed=42)

    dataset = DatasetDict(
        {
            'train': merged_train,
            'validation': merged_valid,
            'test': merged_test
        }
    )

    if model_args.use_encoder_decoder:
        tokenizer = AutoTokenizer.from_pretrained(decoder)
        feature_extractor = AutoFeatureExtractor.from_pretrained(encoder)
        processor = TrOCRProcessor(feature_extractor, tokenizer)
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder, decoder)

    else:
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)

    fn_kwargs = dict(
        processor=processor,
    )
    df = dataset.map(preprocess, fn_kwargs=fn_kwargs, remove_columns=["id"])

    # split dataset into train and test
    train_dataset = df['train']
    eval_dataset = df['validation']
    predict_dataset = df['test']
    df_pred = pd.DataFrame(dataset['test'])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Predict dataset size: {len(predict_dataset)}")

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # set special tokens used for creating the decoder_input_ids from the labels

    cer_metric = load_metric("cer.py")
    wer_metric = load_metric("wer.py")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(
            labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer, "wer": wer}

    def table_results(dataset_pred, save_dir):
        def image_base64(im):
            with BytesIO() as buffer:
                im.save(buffer, 'jpeg')
                return base64.b64encode(buffer.getvalue()).decode()

        def image_formatter(im):
            return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

        ht = dataset_pred.to_html(
            formatters={'image': image_formatter}, escape=False)
        text_file = open(save_dir+"/"+"pred.html", "w")
        text_file.write(ht)
        text_file.close()

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Training model")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("Evaluating model")
    metrics = trainer.evaluate(metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print("Predicting")
    predict_results = trainer.predict(
        predict_dataset,
        metric_key_prefix="predict",
    )
    metrics = predict_results.metrics
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    predictions = processor.batch_decode(
        predict_results.predictions,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    predictions = [pred.strip() for pred in predictions]
    df_pred["predictions"] = predictions

    save_dir = "{}/{}-{}-{}".format(train_args.output_dir,
                                    data_args.dataset_config_name,
                                    decoder_split,
                                    encoder_split)

    os.makedirs(save_dir, exist_ok=True)
    table_results(df_pred, save_dir)
    processor.save_pretrained(save_dir)
    trainer.create_model_card(save_dir)
    trainer.save_model(save_dir)


if __name__ == "__main__":
    main()

from datasets import load_dataset
import fire
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, EarlyStoppingCallback
import os
import sys
sys.path.append("/home/gagan/lab/arocr/OCR/arocr_v2")

from dataset import OCRDataset
from get_model import get_model
from metrics import Metrics


def run(
        run_name='debug',
        encoder_name='facebook/deit-tiny-patch16-224',
        decoder_name='asafaya/bert-mini-arabic',
        max_len=300,
        num_decoder_layers=2,
        output_dir='/home/gagan30/scratch/arocr/output',
):
    dataset = load_dataset("gagan3012/OnlineKhatt")

    model, processor = get_model(encoder_name, decoder_name, max_len, num_decoder_layers)

    # keep package 0 for validation
    
    train_dataset = OCRDataset(dataset, processor, 'train', max_len, augment=True, skip_packages=[0])
    eval_dataset = OCRDataset(dataset, processor, 'dev', max_len, augment=False, skip_packages=[0])
    pred_dataset = OCRDataset(dataset, processor, 'test', max_len, augment=False, skip_packages=range(1, 9999))

    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        num_train_epochs=10,
        weight_decay=0.005,
        learning_rate=5e-5,
        seed=42,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        do_train=True,
        do_eval=True,
        do_predict=True,
        output_dir=output_dir,
        run_name=run_name,
        overwrite_output_dir=True,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )
    print("Starting training...")
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
        pred_dataset,
        metric_key_prefix="predict",
    )
    metrics = predict_results.metrics
    predictions = processor.batch_decode(
        predict_results.predictions,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    predictions = [pred.strip() for pred in predictions]
    output_prediction_file = os.path.join(
        output_dir, "generated_predictions.txt"
    )
    print(predictions)
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(predictions))
    wandb.finish()


if __name__ == '__main__':
    fire.Fire(run)

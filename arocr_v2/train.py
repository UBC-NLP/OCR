from datasets import load_dataset
import fire
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, EarlyStoppingCallback
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.append("/home/gagan/lab/arocr/OCR/arocr_v2")

from dataset import OCRDataset
from get_model import get_model
from metrics import Metrics
from utils import tensor_to_image


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

    print("train_dataset:", train_dataset)
    print("eval_dataset:", eval_dataset)
    print("pred_dataset:", pred_dataset)

    for i in range(5):
        sample = train_dataset[i]
        img = tensor_to_image(sample['pixel_values'])
        tokens = sample['labels']
        tokens[tokens == -100] = processor.tokenizer.pad_token_id
        text = ''.join(processor.decode(tokens, skip_special_tokens=True).split())

        print(f'{i}:\n{text}\n')
        plt.imshow(img)
        plt.show()

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=default_data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=default_data_collator)
    pred_dataloader = DataLoader(pred_dataset, batch_size=32, shuffle=False, collate_fn=default_data_collator)

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
        #report_to="wandb",
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
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
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
        pred_dataloader,
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

if __name__ == '__main__':
    fire.Fire(run)

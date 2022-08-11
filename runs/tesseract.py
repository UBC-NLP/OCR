from datasets import load_dataset
import pandas as pd
from datasets import load_metric
import pytesseract


def main(dataset):
    dataset = load_dataset(
        "/home/gagan30/scratch/arocr/AraOCR_dataset", dataset)

    df_pred = pd.DataFrame(dataset['test'])
    cer = load_metric("cer.py")
    wer = load_metric("wer.py")

    custom_config = r'--oem 3 --psm 6'

    output_list = []
    for img in df_pred['image'].to_list():
        output = pytesseract.image_to_string(
            img.convert('RGB'), config=custom_config, lang='ara')
        output_list.append(output)

    df_pred['pred'] = output_list
    cer_score = cer.compute(
        predictions=df_pred['pred'].to_list(), references=df_pred['text'].to_list())
    wer_score = wer.compute(
        predictions=df_pred['pred'].to_list(), references=df_pred['text'].to_list())

    print("Dataset: {}".format(dataset))
    print("Char Error Rate: {}".format(cer_score))
    print("Word Error Rate: {}".format(wer_score))

if __name__ == "__main__":
    datasets = ['PATS01', 'IDPL-PFOD', 'OnlineKhatt',
                'ADAB', 'alexuw', 'MADBase', 'AHCD']
    for i in datasets:
        main(i)

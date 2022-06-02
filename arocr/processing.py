# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to African NLU Dataset"""
import csv
import json
import datasets
import textwrap


logger = datasets.logging.get_logger(__name__)

dataset_name = "Arabic OCR dataset benchmark"

_MAIN_CITATION = """\
WRITE CITATION
"""

_MAIN_DESCRIPTION = """\
WRITE DESCRIPTION
"""

_URL = "/project/6007993/DataBank/OCR_data/Datasets/al/_Ready/AraOCR_dataset/data"
_TASKS = [
    # line based
    "PATS01",
    "IDPL-PFOD",
    "UPTI",
    "OnlineKHATT",
    # word based
    "ADAB",
    "alexuw",
    "shotor",
    # char based
    "MADBase",
    "AHCD"
]


# _TASKS_data={
#         #(1) Line-based datasets (image, text)
#         'OnlineKhatt':f'{_URL}/OnlineKhatt/',


# }


class AraOCR_dataset_config(datasets.BuilderConfig):
    """BuilderConfig for Arabic OCR Benchmark"""

    def __init__(
        self,
        text_features,
        label_column,
        label_classes,
        citation,
        **kwargs,
    ):
        super(AraOCR_dataset_config, self).__init__(**kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.citation = citation


class AraOCR_dataset(datasets.GeneratorBasedBuilder):
    """AraOCR_dataset datasets."""

    _VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = AraOCR_dataset_config

    BUILDER_CONFIGS = []

    # create text classification single input tasksconfig
    BUILDER_CONFIGS.extend(
        [
            AraOCR_dataset_config(
                name=task,
                version=datasets.Version("1.0.0"),
                description=_MAIN_DESCRIPTION,
                text_features={"id": 0, "image": "image"},
                label_classes=None,
                label_column="text",
                citation=_MAIN_CITATION,
            )
            for task in _TASKS
        ]
    )

    def _info(self):
        features = {
            text_feature: datasets.Value("string")
            for text_feature in self.config.text_features.keys()
        }
        features[self.config.label_column] = datasets.Value("string")
        features["id"] = datasets.Value("int32")

        return datasets.DatasetInfo(
            description=_MAIN_DESCRIPTION,
            features=datasets.Features(features),
            citation=self.config.citation,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}/{self.config.name}/train.tsv",
            "valid": f"{_URL}/{self.config.name}/valid.tsv",
            "test": f"{_URL}/{self.config.name}/test.tsv",
        }
        # download fron the original compressed file(s) and save it in cache folder
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["valid"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                },
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating {} examples from = {}".format(dataset_name, filepath))
        """Yields examples."""
        # if self.config.name in ['topic-light']:
        #     csv.field_size_limit(sys.maxsize)
        content_col = "image"
        label_col = "text"
        with open(filepath, encoding="utf-8") as f:
            data = csv.DictReader(f, delimiter="\t", quotechar='"')
            for row_id, row in enumerate(data):
                # print (row)
                yield row_id, {
                    "id": row_id,
                    content_col: row[content_col],
                    label_col: row[label_col],
                }

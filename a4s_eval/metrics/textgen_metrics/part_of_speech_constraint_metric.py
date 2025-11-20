"""
The implementation of the metric for the part of speech constraint in a
textattack library.
"""
from datetime import datetime

import stanza
from datasets import load_dataset
from textattack import Attack
from textattack.search_methods import SearchMethod
from textattack.shared.validators import goal_function
from textattack.transformations import Transformation

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.textgen_metric_registry import TextgenMetric
from a4s_eval.service.functional_model import TextGenerationModel
from collections import defaultdict


class PartOfSpeechConstraintMetric(TextgenMetric):
    def __call__(
            self,
            datashape: DataShape,
            model: Model,
            dataset: Dataset,
            functional_model: TextGenerationModel,
    ) -> list[Measure]:
        print("Measure the metric here....")
        stanza.download('en')



        # Involves taking the original text and using a POS tagger,
        # for example Stanford POS Tagger. We need to identify each word and
        # check any changes to the word still match the tag of the original
        # word. If the drift is too large, we will reject it as being an
        # attack. Examples this would check to make sure nouns are replaced
        # with nouns, and not an adjective.

        self._nlp = stanza.Pipeline('en', processors='tokenize,pos')

        tagged_data = []
        for i, example in enumerate(dataset.select(range(5))):
            text = example["text"]
            tagged = self.pos_tag_text(text)
            tagged_data.append(tagged)

        self.evaluate_pos_tags(tagged_data)


        items = []
        measure = Measure(name="part_of_speech_constraint_metric",
                          score=0.0, time=datetime.now())
        items.append(measure)
        my_search_method = PartOfSpeechSearchMethod()
        my_transformation = PartOfSpeechTransformation()
        my_attack = Attack(goal_function=goal_function,
                           transformation=my_transformation,
                           constraints=[],
                           search_method=my_search_method)
        return items

    def pos_tag_text(self, text):
        doc = self._nlp(text)
        return [(word.text, word.upos) for sent in doc.sentences for word in sent.words]

    def evaluate_pos_tags(self, tagged_reviews):
        tag_counts = defaultdict(int)
        total_tags = 0

        for review_tags in tagged_reviews:
            for word, tag in review_tags:
                tag_counts[tag] += 1
                total_tags += 1

        # Print tag distribution
        for tag, count in tag_counts.items():
            print(f"{tag}: {count} ({count / total_tags:.2%})")

class PartOfSpeechSearchMethod(SearchMethod):
    def perform_search(self, initial_result):
        pass


class PartOfSpeechTransformation(Transformation):
    def _get_transformations(self, current_text, indices_to_modify):
        pass

    def perform_transformation(self, result):
        pass

    def perform_goal_function(self, result):
        pass

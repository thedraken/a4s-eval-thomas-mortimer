"""
The implementation of the metric for the part of speech constraint in a
textattack library.
"""
from datetime import datetime

from textattack import Attack
from textattack.search_methods import SearchMethod
from textattack.shared.validators import goal_function
from textattack.transformations import Transformation

from a4s_eval.data_model.evaluation import DataShape, Model, Dataset
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import ModelMetric
from a4s_eval.service.functional_model import TabularClassificationModel

import nltk




class PartOfSpeechConstraintMetric(ModelMetric):
    def __call__(
            self,
            datashape: DataShape,
            model: Model,
            dataset: Dataset,
            functional_model: TabularClassificationModel,
    ) -> list[Measure]:
        print("Measure the metric here....")
        nltk.download('stanford-postagger')
        # Involves taking the original text and using a POS tagger,
        # for example Stanford POS Tagger. We need to identify each word and
        # check any changes to the word still match the tag of the original
        # word. If the drift is too large, we will reject it as being an
        # attack. Examples this would check to make sure nouns are replaced
        # with nouns, and not an adjective.
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

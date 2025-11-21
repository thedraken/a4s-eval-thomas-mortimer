"""
The implementation of the metric for the part of speech constraint in a
textattack library.
"""
from datetime import datetime
from typing import List

import stanza

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.textgen_metric_registry import TextgenMetric
from a4s_eval.service.functional_model import TextGenerationModel
from a4s_eval.utils.logging import get_logger


class PartOfSpeechConstraintMetric(TextgenMetric):
    """
    Involves taking the original text and using a POS tagger,
    for example Stanford POS Tagger. We need to identify each word and
    check any changes to the word still match the tag of the original
    word. If the drift is too large, we will reject it as being an
    attack. Examples this would check to make sure nouns are replaced
    with nouns, and not an adjective.
    """
    def __call__(
            self,
            datashape: DataShape,
            model: Model,
            dataset: Dataset,
            functional_model: TextGenerationModel,
    ) -> list[Measure]:
        print("Measure the metric here....")
        stanza.download('en')


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

        """
        Measure the POS constraint metric for the dataset.
        """
        items = []
        total_score = 0.0
        num_samples = 0

        for i, (original_text, label) in enumerate(dataset):
            # Tag the original text
            original_tags = self.pos_tag_text(original_text)

            # Simulate a perturbed text
            # TODO implement attack logic
            perturbed_text = original_text
            perturbed_tags = self.pos_tag_text(perturbed_text)

            # Calculate the POS tag match score
            score = self.evaluate_pos_tags(original_tags, perturbed_tags)
            total_score += score
            num_samples += 1

            # Create a measure for this sample
            measure = Measure(
                name="part_of_speech_constraint_metric",
                score=score,
                time=datetime.now(),
            )
            items.append(measure)

        # Calculate the average score
        avg_score = total_score / num_samples if num_samples > 0 else 0.0

        # Create a summary measure
        summary_measure = Measure(
            name="part_of_speech_constraint_metric_avg",
            score=avg_score,
            time=datetime.now(),
        )
        items.append(summary_measure)

        return items

    def pos_tag_text(self, text: str) -> List[tuple]:
        """Tag a text with POS tags using stanza."""
        doc = self._nlp(text)
        return [(word.text, word.upos) for sent in doc.sentences for word in sent.words]

    def evaluate_pos_tags(self, original_tags: List[tuple], perturbed_tags: List[tuple]) -> float:
        """
        Evaluate the POS tag drift between original and perturbed text.
        Returns a score between 0 and 1, where 1 means no drift.
        """
        if len(original_tags) != len(perturbed_tags):
            return 0.0

        matches = 0
        for (orig_word, orig_tag), (pert_word, pert_tag) in zip(original_tags, perturbed_tags):
            if orig_tag == pert_tag:
                matches += 1

        length_of_original_tags = len(original_tags)
        if length_of_original_tags != 0:
            return matches / length_of_original_tags
        logger = get_logger()
        logger.warning("No original tags found, returning 0.0")
        return 0.0
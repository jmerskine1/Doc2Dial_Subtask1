import json
import argparse

from datasets import load_dataset
from datasets import load_metric


def sharedtask1_metrics(prediction_json, split, cache_dir=None):
    metric = load_metric("squad_v2")

    predictions = json.load(open(prediction_json, "r"))
    d_id_prediction = {}
    for ele in predictions:
        d_id_prediction[ele["id"]] = 0

    references = []
    d_id_reference = {}
    dataset = load_dataset(
        "doc2dial",
        name="doc2dial_rc",
        split=split,
        ignore_verifications=True,
        cache_dir=cache_dir,
    )
    for ex in dataset:
        if ex["id"] not in d_id_prediction:
            continue
        d_id_reference[ex["id"]] = 0
        references.append(
            {
                "id": ex["id"],
                "answers": ex["answers"],
            }
        )
    assert (
        len(predictions)
        == len(references)
        == len(d_id_prediction)
        == len(d_id_reference)
    ), "Ensure the matching count of instances of references and predictioins"

    metric.add_batch(predictions=predictions, references=references)
    final_score = metric.compute()
    """
    print(final_score)
    OrderedDict([('exact', 33.333333333333336), ('f1', 38.095238095238095), ('span', 33.333333333333336), ('total', 3), ('HasAns_exact', 33.333333333333336), ('HasAns_f1', 38.095238095238095), ('HasAns_total', 3)])
    """
    return final_score

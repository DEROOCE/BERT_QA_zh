import json 
import os 
import collections
import numpy as np
from datasets import Dataset 
from torch.utils.data import DataLoader

def generate_dataset(path): 
    ids, titles, contexts, questions, answers = [], [], [], [], [] 

    for line in open(path, 'r', encoding='utf8'):
        ids.append(json.loads(line)["id"])
        titles.append(json.loads(line)["title"])
        contexts.append(json.loads(line)["context"])
        questions.append(json.loads(line)["question"])
        answers.append(json.loads(line)["answers"])
        
    data = {
        "id": ids,
        "title": titles,
        "context": contexts,
        "question": questions,
        "answers": answers
    }
    dataset = Dataset.from_dict(data)
    return dataset # torch dataset object

def generate_dataloader(data, prepare_fn, args, tokenizer, collate_fn, 
                        shuffle: bool = False,
                        drop_last: bool =False):
    dataset = data.map(
        prepare_fn,
        batched=True,
        remove_columns=data.column_names,
        fn_kwargs={"args": args, "tokenizer": tokenizer}
    )
    batch_size = args.per_gpu_batch_size * args.n_gpus
    data_iter = DataLoader(dataset, 
                            shuffle=shuffle,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            num_workers=4,
                            drop_last=drop_last)
    return data_iter

def prepare_train_features(examples, tokenizer=None, args=None):
    # 既要对examples进行truncation（截断）和padding（补全）还要还要保留所有信息，所以要用的切片的方法。
    # 每一个一个超长文本example会被切片成多个输入，相邻两个输入之间会有交集。
    tokenized_examples = tokenizer(
        examples["question" if args.pad_on_right else "context"],
        examples["context" if args.pad_on_right else "question"],
        truncation="only_second" if args.pad_on_right else "only_first",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 我们使用overflow_to_sample_mapping参数来映射切片片ID到原始ID。
    # 比如有2个expamples被切成4片，那么对应是[0, 0, 1, 1]，前两片对应原来的第一个example。
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # offset_mapping也对应4片
    # offset_mapping参数帮助我们映射到原始输入，由于答案标注在原始输入上，所以有助于我们找到答案的起始和结束位置。
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 重新标注数据
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 对每一片进行处理
        # 将无答案的样本标注到CLS上
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 区分question和context
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 拿到原始的example 下标.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # 如果没有答案，则使用CLS所在的位置为答案.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 答案的character级别Start/end位置.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # 找到token级别的index start.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if args.pad_on_right else 0):
                token_start_index += 1

            # 找到token级别的index end.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if args.pad_on_right else 0):
                token_end_index -= 1

            # 检测答案是否超出文本长度，超出的话也适用CLS index作为标注.
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 如果不超出则找到答案token的start和end位置。.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_validation_features(examples, tokenizer=None, args=None):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if args.pad_on_right else "context"],
        examples["context" if args.pad_on_right else "question"],
        truncation="only_second" if args.pad_on_right else "only_first",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if args.pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(args, 
                            tokenizer,
                            examples, 
                            features, 
                            raw_predictions):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index].cpu().numpy()
            end_logits = all_end_logits[feature_index].cpu().numpy()
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -args.n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -args.n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > args.max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not args.squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions


if __name__ == "__main__":
    pass
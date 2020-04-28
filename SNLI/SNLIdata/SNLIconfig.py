class filenames:
    label2id = 'SNLI_label2id.json'
    train_set = "SNLI_train_ids.json"
    valid_set = "SNLI_dev_ids.json"
    test_set = "SNLI_test_ids.json"
    id2content = "SNLI_input.json"
    model_name = 'model.pt'


class parameters:
    max_def_lens = 64  # append one [SEP] at the end of the sentence
    batch_size = 16
    random_seed = 2019
    epoches = 20
    BERT_MODEL = "bert-base-uncased"

    def __str__(self):
        for attr, value in self.__dict__.items():
            print(attr, value)

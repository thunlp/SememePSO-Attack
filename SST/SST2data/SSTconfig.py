class filenames:
    label2id = 'SST_label2id.json'
    train_set = "SST_train_ids.json"
    valid_set = "SST_dev_ids.json"
    test_set = "SST_test_ids.json"
    id2content = "SST_input.json"
    model_name = 'model.pt'


class parameters:
    max_def_lens = 32  # append one [SEP] at the end of the sentence
    batch_size = 8
    random_seed = 2019
    epoches = 60
    BERT_MODEL = "bert-base-uncased"

    def __str__(self):
        for attr, value in self.__dict__.items():
            print(attr, value)

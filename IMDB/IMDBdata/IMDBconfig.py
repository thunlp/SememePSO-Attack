class filenames:
    label2id = 'IMDB_label2id.json'
    train_set = "IMDB_train_ids.json"
    valid_set = "IMDB_dev_ids.json"
    test_set = "IMDB_test_ids.json"
    id2content = "IMDB_input.json"
    model_name = 'model.pt'


class parameters:
    max_def_lens = 254  # append one [SEP] at the end of the sentence
    batch_size = 8
    random_seed = 2018
    epoches = 60
    BERT_MODEL = "bert-base-uncased"

    def __str__(self):
        for attr, value in self.__dict__.items():
            print(attr, value)

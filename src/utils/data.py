from datasets import load_dataset

def loading_data(dataset_name: str, num_data: int):
    data = load_dataset(dataset_name, split='train')
    data = data.select(range(num_data))
    data = data.train_test_split(0.2)
    train_data = data['train']
    test_data = data['test']
    return train_data, test_data



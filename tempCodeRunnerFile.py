
def load_and_preprocess_data(data_directory):
    emails = []
    for filename in os.listdir(data_directory):
        with open(os.path.join(data_directory, filename), 'r', encoding='utf-8') as file:
            emails.append(file.read())
    return emails
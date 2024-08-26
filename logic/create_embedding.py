import logic.embeds as embeds


def create_embedding_codebert(code_path, file_name, batch_size=64):
    embeds.build_codebert(code_path, file_name, batch_size)


def create_embedding_graphcodebert(code_path, file_name, batch_size=64):
    embeds.build_graphcodebert(code_path, file_name, batch_size)


def create_embedding_bert_nli_mean(code_path, file_name, batch_size=64):
    embeds.build_bert_nli_mean(code_path, file_name, batch_size)

if __name__ == "__main__":
    create_embedding_codebert("../data/raw/unique_data_setV3.json", "temp", 64)

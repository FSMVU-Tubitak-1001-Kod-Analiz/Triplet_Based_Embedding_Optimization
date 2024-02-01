import logic.embeds as embeds


def create_embedding_codebert(code_path, file_name, batch_size=64):
    embeds.build_codebert(code_path, file_name, batch_size)


def create_embedding_graphcodebert(code_path, file_name, batch_size=64):
    embeds.build_graphcodebert(code_path, file_name, batch_size)


def create_embedding_bert_nli_mean(code_path, file_name, batch_size=64):
    embeds.build_bert_nli_mean(code_path, file_name, batch_size)
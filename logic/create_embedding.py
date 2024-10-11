import logic.embeds as embeds
from logic.embeds import build_tokens_bert_nli_mean, build_tokens_codebert, build_tokens_graphcodebert

# def build_token_from_text(code, method):
#     with tempfile.NamedTemporaryFile("w", suffix=".json") as tmp:
#         code = r'{"function":"protected ChannelFuture sendMapOutput(ChannelHandlerContext ctx, Channel ch, String user, String mapId, int reduce, MapOutputInfo mapOutputInfo) throws IOException{\n    final TezIndexRecord info = mapOutputInfo.indexRecord;\n    final ShuffleHeader header = new ShuffleHeader(mapId, info.getPartLength(), info.getRawLength(), reduce);\n    final DataOutputBuffer dob = new DataOutputBuffer();\n    header.write(dob);\n    ch.write(wrappedBuffer(dob.getData(), 0, dob.getLength()));\n    final File spillfile = new File(mapOutputInfo.mapOutputFileName.toString());\n    RandomAccessFile spill;\n    try {\n        spill = SecureIOUtils.openForRandomRead(spillfile, \"r\", user, null);\n    } catch (FileNotFoundException e) {\n        LOG.info(spillfile + \" not found\");\n        return null;\n    }\n    ChannelFuture writeFuture;\n    if (ch.pipeline().get(SslHandler.class) == null) {\n        boolean canEvictAfterTransfer = true;\n        if (!shouldAlwaysEvictOsCache) {\n            canEvictAfterTransfer = (reduce > 0);\n        }\n        final FadvisedFileRegion partition = new FadvisedFileRegion(spill, info.getStartOffset(), info.getPartLength(), manageOsCache, readaheadLength, readaheadPool, spillfile.getAbsolutePath(), shuffleBufferSize, shuffleTransferToAllowed, canEvictAfterTransfer);\n        writeFuture = ch.write(partition);\n    } else {\n        final FadvisedChunkedFile chunk = new FadvisedChunkedFile(spill, info.getStartOffset(), info.getPartLength(), sslFileBufferSize, manageOsCache, readaheadLength, readaheadPool, spillfile.getAbsolutePath());\n        writeFuture = ch.write(chunk);\n    }\n    return writeFuture;\n}","smellKey":"java:S1172","smellId":null}'
#         print(code)
#     tmp.write(code)
#     tmp.flush()
#     inputs = create_embedding.build_tokens_graphcodebert(tmp.name)


def create_embedding_codebert(code_path, file_name, batch_size=64):
    embeds.build_codebert(code_path, file_name, batch_size)


def create_embedding_graphcodebert(code_path, file_name, batch_size=64):
    embeds.build_graphcodebert(code_path, file_name, batch_size)


def create_embedding_bert_nli_mean(code_path, file_name, batch_size=64):
    embeds.build_bert_nli_mean(code_path, file_name, batch_size)


if __name__ == "__main__":
    create_embedding_codebert("../data/raw/unique_data_setV3.json", "temp", 64)

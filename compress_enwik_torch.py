import numpy as np
import torch
import tqdm

from collections.abc import Iterator
from typing import Union

import constants
import data_loaders
import utils
import arithmetic_coder

from train_enwik_torch import TransformerConfig, TransformerDecoder

def predict_fn(model, tokenized_data_batch):
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Convert numpy array to PyTorch tensor
        x = torch.tensor(tokenized_data_batch, dtype=torch.int64).to('cuda')
        # Get logits from the model        
        logits = model(x)
        # Convert to log probabilities and then to numpy
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
    return log_probs

def load_model(path):
    config = TransformerConfig(vocab_size=constants.ALPHABET_SIZE)
    model = TransformerDecoder(config)
    params = torch.load(path)
    model.load_state_dict(params)
    model.to('cuda')
    return model


def llm_compress(tokenized_data, model, use_slow_lossless_compression=True):
    if use_slow_lossless_compression:
        log_probs = list()
        for t in range(len(tokenized_data)):
            # assume tokenized_data = [h,e,l,l,o]
            # t0: input = [h] 實際模型的輸入是 [BOS]，因為還要做 right_shift,模型的輸出是 p(h | <bos>)
            # t1: input = [h,e] 實際模型的輸入是 [h]，因為還要做 right_shift，模型的輸出是 p(e | h)
            # t2: input = [h,e,l]
            # t3: input = [h,e,l,l]
            # t4: input = [h,e,l,l,o]
            # 为什么这里必须一步步来，不能一次输入所有的 tokenized_data? 因为解码也是一个一个token解码的。
            # 直接一次 forward "<bos>hell" 在 hell位置上得到的 logits 其实跟一步步得到的logits不同
            # 为什么不同？因为 casual mask 只是应用在了 attention 的计算里，而整个transformer 的计算组件里还有很多
            # 没有用到 casual mask 的组件，比如 layer normalization 和 FFN 等。
            input_seq = tokenized_data[None, : t + 1]
            subsequence_probs = predict_fn(model, input_seq)
            last_token_probs = subsequence_probs[0, -1]
            log_probs.append(last_token_probs)
        log_probs = np.vstack(log_probs)
    else:
        log_probs = predict_fn(model, tokenized_data[None])[0, ...]
    probs = np.exp(log_probs)
    
    output = list()
    encoder = arithmetic_coder.Encoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        output_fn=output.append,
    )

    for pdf, symbol in zip(probs, tokenized_data):
        encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)
    encoder.terminate()
    compressed_bits = ''.join(map(str, output)) 
    # 假设压缩后的bits 比如 8519 bits， 因為 8519 不是 8 的倍數，所以需要 padding
    # 最少需要在左邊pad一個bit-0，变成 8520 bits，这样的话 num_padded_bits = 1
    compressed_data, num_padded_bits = utils.bits_to_bytes(compressed_bits)
    return compressed_data, num_padded_bits


def llm_decompress(model, compressed_data, num_padded_bits, model_seq_len = constants.CHUNK_SIZE_BYTES):
    data_iter = iter(utils.bytes_to_bits(compressed_data, num_padded_bits=num_padded_bits))
    # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
    # from the compressed input and returns `None` when the input is exhausted.
    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> Union[int, None]:
      try:
        return int(next(bit_sequence))
      except StopIteration:
        return None

    decoder = arithmetic_coder.Decoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        input_fn=_input_fn,
    )

    # We need a dummy token because the language model right-shifts the sequence
    # by one when computing the conditional probabilities. Concretely, at every
    # step, we need the `pdf` of the next token given all currently decompressed
    # tokens, but without a dummy token, the last `pdf` would be that of the last
    # already decompressed token. The value of the dummy token is irrelevant.
    sequence_array = np.empty((1,), dtype=np.uint8)
    probs = np.exp(predict_fn(model, sequence_array[None])[0, ...])

    uncompressed_length = model_seq_len
    for idx in range(uncompressed_length):
      token = decoder.decode(
          utils.normalize_pdf_for_arithmetic_coding(probs[idx])
      )
      sequence_array = np.insert(sequence_array, -1, token)
      if len(sequence_array) > model_seq_len:
        break
      probs = np.exp(predict_fn(model, sequence_array[None])[0, ...])  
    decoded_data = sequence_array[:-1].tobytes()
    return decoded_data

if __name__ == '__main__':
    # Load model
    model = load_model('params.pth')
    use_slow_lossless_compression = True
    use_tqdm = True
    num_chunks = 1

    # Prepare data to be compressed
    enwik9_data_generator = data_loaders.get_enwik9_iterator(
          num_chunks=num_chunks,
          chunk_start_idx= constants.NUM_CHUNKS // 10,
          sequence_length=constants.CHUNK_SIZE_BYTES,
    )
    if use_tqdm:
        enwik9_data_generator = tqdm.tqdm(enwik9_data_generator, total=num_chunks)

    # Initialization
    mask_fn = utils.zero_most_significant_bit_if_not_ascii_decodable
    num_missed_bits = running_time = raw_length = compressed_length = 0
    print("NUM_MISSED_BITS: ",num_missed_bits)
    
    # Start compress chunk by chunk
    print("Start compress chunk by chunk!")
    for rawdata in enwik9_data_generator:
        data_to_encode, missed_bits = mask_fn(rawdata) # 大模型压缩我们只压缩 ASCII 编码表里的字符，超出这个之外的就不能了。
        num_missed_bits += missed_bits
        tokenized_data = np.frombuffer(data_to_encode, dtype=np.uint8) # 对于文本编码，我们直接把每个字符的 ASCII 码作为 token

        compressed_data, num_padded_bits = llm_compress(tokenized_data, model)
        raw_length += len(data_to_encode)
        compressed_length += len(compressed_data)

        if use_slow_lossless_compression: # 解码一下来验证一下是否能够恢复原始数据
            decoded_data = llm_decompress(model, compressed_data, num_padded_bits)
            if data_to_encode == decoded_data:
                print('SUCCESS: Data was successfully compressed and decompressed!')
            else:
                print('FATAL ERROR: Decompressed data does not match original data!')
                break
    
    # Since language models are trained on ASCII strings, they cannot handle all
    # byte values. Thus, we mask the data to be ASCII-decodable by zeroing
    # `num_missed_bits` of the most significant bits. However, this means that we
    # are effectively only compressing `num_bits - num_missed_bits` bits, so we
    # rescale the `compressed_length` to account for this.
    total_num_bits = 8 * num_chunks * constants.CHUNK_SIZE_BYTES
    scale_factor = total_num_bits / (total_num_bits - num_missed_bits)
    compressed_length *= scale_factor

    # Calculate compression ratio
    compression_ratio = compressed_length / raw_length
    print(f'Compression ratio (Chunked): {compression_ratio:.4f}')
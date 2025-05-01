import argparse
import os
import datasets
from transformers import AutoTokenizer
import torch # 用于获取 pad_token_id
import re
import glob

# 设置 TOKENIZERS_PARALLELISM 环境变量，避免 Tokenizer 库产生警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize_function(examples, tokenizer, max_length, eos_token):
    """
    Tokenization function to be applied to the dataset batches.
    Adds EOS token, tokenizes, pads/truncates, and calculates loss mask.
    """
    try:
        # 使用正则表达式移除无法识别的字符
        def clean_text(text):
            return re.sub(r'[^\x00-\x7F]+', '', text)

        # 将 EOS token 添加到文本末尾
        texts_with_eos = [clean_text(str(text)) + eos_token for text in examples['text']]

        # 进行 Tokenize 操作
        encoding = tokenizer(
            texts_with_eos,
            max_length=max_length,
            padding='max_length', # 直接填充到 max_length
            truncation=True,
            return_tensors=None, # datasets 的 map 默认处理 Python list
            add_special_tokens=False # 通常在 apply_chat_template 或手动添加特殊 token 时控制
        )

        # 获取 pad_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # 计算 loss_mask (在 Y 上计算损失的部分)
        # loss_mask 应该标记所有非 padding 的 token
        loss_masks = []
        for input_ids in encoding['input_ids']:
            # loss_mask 初始化为全 0
            mask = [0] * len(input_ids)
            # 将非 padding token 的位置标记为 1
            for i, token_id in enumerate(input_ids):
                if token_id != pad_token_id:
                    mask[i] = 1
            # loss_mask 需要向右移动一位，与 Y 对齐
            # 所以我们直接使用原始 mask，在 Dataset 的 __getitem__ 中再处理移位
            loss_masks.append(mask)

        # 返回包含 input_ids 和 loss_mask 的字典
        # 注意：datasets 的 map 函数期望返回一个字典，key 是列名
        return {
            'input_ids': encoding['input_ids'],
            'loss_mask': loss_masks # 保存原始的 mask
        }
    except Exception as e:
        print(f"Error during tokenization: {e}")
        #! 处理异常，例如跳过无法处理的文本或记录日志
        return {'input_ids': [], 'loss_mask': []}

def main():
    parser = argparse.ArgumentParser(description="Preprocess pretraining data using Hugging Face datasets.")
    parser.add_argument("--data_path", type=str, default="/home/ytllm/.cache/modelscope/datasets/BAAI/Aquila-135M-Datasets/pretrain/wiki", help="Path to the directory containing .parquet files.")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Name or path of the tokenizer.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument("--output_dir", type=str, default="/home/ytllm/.cache/pretrain_data/", help="Directory to save the processed dataset.")
    parser.add_argument("--num_proc", type=int, default=40, help="Number of processes to use for map function. Defaults to cpu count.")

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer_name}")
    # 加载 Tokenizer
    # trust_remote_code=True 对于某些模型是必要的，比如 Qwen
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    # 确保 tokenizer 有 pad token，如果没有，通常设置为 eos_token
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Looking for .parquet files in: {args.data_path}")
    # 使用 glob 查找所有 .parquet 文件
    parquet_files = list(glob.glob(os.path.join(args.data_path, '**/*.parquet'), recursive=True))
    if not parquet_files:
        raise ValueError(f"No .parquet files found in {args.data_path}")
    print(f"Found .parquet files: {parquet_files}")
    raw_datasets = datasets.load_dataset('parquet', data_files=parquet_files)

    # 如果数据集有不同的 split (如 'train', 'validation')，你需要选择一个，或者都处理
    # 这里假设只有一个 'train' split，或者将所有文件合并加载
    if 'train' not in raw_datasets:
        # 如果没有明确的 'train' split，可能是因为 load_dataset 将所有文件视为一个 split
        # 通常这个 split 会被命名为 'train'，但如果不是，我们需要获取实际的 split 名称
        split_name = list(raw_datasets.keys())[0]
        print(f"Using split '{split_name}' from the loaded dataset.")
        dataset_split = raw_datasets[split_name]
    else:
        dataset_split = raw_datasets['train']

    # print(f"Found {len(dataset_split)} examples.")
    print("Starting tokenization...")

    # 获取 EOS token
    eos_token = tokenizer.eos_token
    if eos_token is None:
        # 对于某些没有明确 eos_token 的模型，可能需要指定一个，或者使用 pad_token
        print("Warning: Tokenizer does not have an EOS token. Using pad_token as EOS.")
        eos_token = tokenizer.pad_token
        if eos_token is None:
             raise ValueError("Tokenizer has neither eos_token nor pad_token defined.")

    # 加载原始数据集后添加以下代码
    total_samples = len(dataset_split)
    subset_size = int(total_samples * 0.2)
    print(f"原始数据集样本数: {total_samples}, 将处理前20%数据: {subset_size} 个样本")
    
    # 创建子集数据集
    subset_dataset = dataset_split.select(range(subset_size))

    # 使用 map 函数进行并行处理
    # batched=True 让 tokenize_function 接收一批数据，效率更高
    # remove_columns 会移除不再需要的原始 'text' 列
    tokenized_dataset = subset_dataset.map(
        tokenize_function,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length, 'eos_token': eos_token},
        batched=True,
        num_proc=args.num_proc, # 使用多进程加速
        remove_columns=dataset_split.column_names # 移除原始列
    )

    print("Tokenization finished.")
    print(f"Saving processed dataset to: {args.output_dir}")

    # 保存处理后的数据集到磁盘 (Arrow 格式)
    tokenized_dataset.save_to_disk(args.output_dir)

    print("Preprocessing complete.")
    print(f"Processed dataset saved in {args.output_dir}")
    print("You can now use this path in the PretrainDataset.")

if __name__ == "__main__":
    main() 
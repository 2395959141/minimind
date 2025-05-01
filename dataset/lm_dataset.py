import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

import glob
import datasets 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#! 直接从处理好的数据集中读取
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.eos_token_id = tokenizer.convert_tokens_to_ids(self.eos_token)
        self.pad_token_id = tokenizer.convert_tokens_to_ids(self.pad_token)
        self.dataset = self.load_data(data_path)

    def load_data(self, data_path):
        # 直接从预处理好的数据集加载
        try:
            dataset = datasets.load_from_disk(data_path)
            print(f"成功加载预处理数据集，包含 {len(dataset)} 个样本")
            return dataset
        except Exception as e:
            raise ValueError(f"无法从 {data_path} 加载预处理数据集: {e}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        
        # 确保 loss_mask 是张量
        if isinstance(sample['loss_mask'], torch.Tensor):
            loss_mask = sample['loss_mask']
        else:
            loss_mask = torch.tensor(sample['loss_mask'], dtype=torch.long)
        
        # 构建训练数据
        X = input_ids[:-1]  # 去掉最后一个 token 作为输入
        Y = input_ids[1:]   # 去掉第一个 token 作为目标
        loss_mask = loss_mask[1:]  # 对齐预测位置
        
        return X, Y, loss_mask

# class PretrainDataset(Dataset):
#     def __init__(self, data_path, tokenizer, max_length=512):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         # self.eos_token = '<|im_end|>'
#         # self.pad_token = '<|endoftext|>'
#         self.eos_token = tokenizer.eos_token
#         self.pad_token = tokenizer.pad_token
#         self.eos_token_id = tokenizer.convert_tokens_to_ids(self.eos_token)
#         self.pad_token_id = tokenizer.convert_tokens_to_ids(self.pad_token)
#         self.samples = self.load_data(data_path)

#     def load_data(self, data_path):
#         #* 递归查找所有子目录下的 .parquet 文件
#         files = []
#         for root, dirs, filenames in os.walk(data_path):
#             for f in filenames:
#                 if f.endswith('.parquet'):
#                     files.append(os.path.join(root, f))
#         if not files:
#             raise FileNotFoundError(f"数据目录 {data_path} 及其子目录下没有找到任何 .parquet 文件！")
#         dfs = []
#         for file in files:
#             try:
#                 df = pd.read_parquet(file)
#                 #! 加载数据集时进行tokenizer操作
#                 df['encoded'] = df['text'].apply(lambda x: self.tokenizer(
#                     str(x) + self.eos_token,
#                     max_length=self.max_length,
#                     padding='max_length',
#                     truncation=True,
#                     return_tensors='pt'
#                 ))
#                 dfs.append(df)
#             except Exception as e:
#                 print(f"读取 {file} 失败: {e}")
#         if not dfs:
#             raise ValueError(f"目录 {data_path} 下的 .parquet 文件全部读取失败！")
#         return pd.concat(dfs, ignore_index=True)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         #sample = self.samples[index]

#         #! 构建输入文本, 末尾加上eos_token
#         #text = str(sample['text']) + self.eos_token
#         # encoding = self.tokenizer(
#         #     text,
#         #     max_length=self.max_length,
#         #     padding='max_length',
#         #     truncation=True,
#         #     return_tensors='pt'
#         # )
#         sample = self.samples[index]
#         encoding = sample['encoded']
#         input_ids = encoding.input_ids.squeeze()
#         loss_mask = (input_ids != self.pad_token_id)

#         X = input_ids[:-1] #* 构建输入数据 X，去掉最后一个 token
#         Y = input_ids[1:]  #* 构建目标数据 Y，去掉第一个 token
#         loss_mask = loss_mask[1:] #! 构建损失掩码，去掉第一个 token
#         return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id =  '<|im_end|>'
        self.eos_id = '<|endoftext|>'
        self.eos_token_id = tokenizer.convert_tokens_to_ids(self.eos_token)
        self.pad_token_id = tokenizer.convert_tokens_to_ids(self.pad_token)

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        ), answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt, answer = self._create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    pass

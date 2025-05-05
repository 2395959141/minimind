import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, max_lr, min_lr=0.0, warmup_init_lr=0.0):
    """
    改进版学习率调度器（线性预热 + 余弦退火）
    参数说明：
    - current_step: 当前训练步数
    - total_steps: 总训练步数
    - max_lr: 最大学习率（目标学习率）
    - min_lr: 最小学习率（默认0）
    - warmup_init_lr: 预热初始学习率（默认0）
    """
    # 安全校验
    if total_steps <= 0:
        return max_lr
    if current_step > total_steps:
        return min_lr
    
    # 计算预热阶段参数
    if args.warmup_steps is not None:  # 手动指定优先
        warmup_steps = args.warmup_steps
    else:  # 未指定时使用比例计算
        warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 确保预热步数不超过总步数
    warmup_steps = min(warmup_steps, total_steps)
    
    # 预热阶段
    if current_step < warmup_steps and warmup_steps > 0:
        return warmup_init_lr + (max_lr - warmup_init_lr) * current_step / warmup_steps
    
    # 余弦退火阶段
    else:
        # 计算退火阶段步数
        decay_steps = total_steps - warmup_steps
        # 当总步数等于预热步数时直接返回最大学习率
        if decay_steps <= 0:
            return max_lr
        
        # 余弦退火计算
        decay_ratio = (current_step - warmup_steps) / decay_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * cosine_decay


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    global_update_step = args.current_step  # 统一使用全局计数器
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(
            current_step=epoch * iter_per_epoch + step,
            total_steps=args.epochs * iter_per_epoch,
            max_lr=args.learning_rate,
            min_lr=args.min_lr,
            warmup_init_lr=args.warmup_init_lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
        
        # 计算实际应记录的步数
        log_interval = args.log_interval
        if args.log_per_updates:
            # 根据实际更新次数计算间隔
            log_interval = max(1, args.log_interval // args.accumulation_steps)
            current_log_step = step
        else:
            current_log_step = step
        
        # 日志记录条件
        if current_log_step % log_interval == 0:
            spend_time = time.time() - start_time
            log_info = [
                f'Epoch:[{epoch+1}/{args.epochs}]',
                f'Batch:[{step}/{iter_per_epoch}]',
                f'Update:[{global_update_step}]' if args.log_per_updates else '',
                f'loss:{loss.item()*args.accumulation_steps:.3f}',
                f'lr:{optimizer.param_groups[-1]["lr"]:.12f}',
                f'Time:{spend_time/(step+1)*iter_per_epoch//60 - spend_time//60}min'
            ]
            Logger(' '.join(filter(None, log_info)))
            
            if wandb and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "global_step": global_update_step
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()

        if (step + 1) % 10000 == 0 and (not ddp or dist.get_rank() == 0):
            test_model_on_prompts(model, tokenizer, args.device, wandb=wandb)
        
        if step % 500 == 0:
            torch.cuda.empty_cache()


def init_model(lm_config):
    if args.use_qwen25_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    else:
        tokenizer = AutoTokenizer.from_pretrained('./model/')
    model = MiniMindForCausalLM(lm_config).to(args.device)

    # 修改检查点路径获取逻辑
    if args.checkpoint_path:  # 手动指定路径优先级最高
        checkpoint_path = args.checkpoint_path
        print(f"使用手动指定的检查点路径: {checkpoint_path}")
    else:  # 保留原有自动生成逻辑
        moe_path = '_moe' if lm_config.use_moe else ''
        checkpoint_path = os.path.join(args.save_dir, f"pretrain_{lm_config.hidden_size}{moe_path}.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        # 新增恢复模式判断
        if args.resume_training:
            # 恢复原有训练进度
            args.current_step = checkpoint.get('current_step', 0)
            args.wandb_step = checkpoint.get('wandb_step', 0)
            args.original_total_steps = checkpoint.get('original_total_steps', args.epochs * iter_per_epoch)
            
            # 恢复优化器状态
            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复梯度缩放器状态
            if 'scaler_state_dict' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            print(f"恢复训练：当前步数{args.current_step}，总步数{args.original_total_steps}")
        else:
            # 重新开始训练设置
            args.current_step = 0
            args.wandb_step = 0
            args.original_total_steps = args.epochs * iter_per_epoch
            print("加载模型权重但重新开始训练")
        
        # 加载模型参数（无论哪种模式都需要加载）
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("未找到检查点，从头开始训练")
        args.current_step = 0
        args.wandb_step = 0

    # 梯度检查点启用
    if args.use_gc:
        model.gradient_checkpointing_enable()
        if not ddp or dist.get_rank() == 0:
            print(f"[优化] 已启用梯度检查点，预计节省显存：{lm_config.hidden_size//1000}GB左右")
    print("词嵌入维度", model.config.vocab_size)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


def test_model_on_prompts(model, tokenizer, device, max_seq_len=80, wandb=None):
    prompts = [
        '帮我写一封感谢信',
        '如何学习深度学习？',
        '用Python实现快速排序',
        '解释Transformer模型原理',
        '鲁迅的《狂人日记》是如何批判封建礼教的？',
        '我咳嗽已经持续了两周，需要去医院检查吗？',
        '详细的介绍光速的物理概念。',
        '推荐一些杭州的特色美食吧。',
        '请为我讲解“大语言模型”这个概念。',
    ]
    model.eval()
    gen_model = model.module if hasattr(model, "module") else model
    
    test_results = {}
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(device)
        with torch.no_grad():
            generated_ids = gen_model.generate(
                inputs["input_ids"],
                max_new_tokens=max_seq_len,
                do_sample=True,
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                top_p=0.85,
                temperature=0.7,
                repetition_penalty=1.1
            )
            response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"\n[测试] 输入: {prompt}\n[输出]: {response}\n")
            test_results[f"测试/{prompt[:10]}"] = response
    
    if wandb:
        wandb.log(test_results)
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100,
                       help="日志间隔（根据log_per_updates决定按批次或更新次数）")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=16, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="/home/ytllm/.cache/modelscope/datasets/gongjy/minimind_dataset/sft_1024.jsonl")
    parser.add_argument("--warmup_steps", type=int, default=None,
                       help="手动指定预热步数（优先于warmup_ratio）")
    parser.add_argument("--use_qwen25_tokenizer", type=bool, default=True)
    parser.add_argument("--use_gc", type=bool, default=False)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="预热阶段比例（当warmup_steps未设置时生效）")
    parser.add_argument("--resume_training", action="store_true",
                       help="从检查点恢复训练")
    parser.add_argument("--current_step", type=int, default=0,
                       help="初始训练步数")
    parser.add_argument("--min_lr", type=float, default=5e-5,
                       help="最小学习率（默认5e-8）")
    parser.add_argument("--warmup_init_lr", type=float, default=5e-8,
                       help="预热初始学习率（默认5e-8）")
    parser.add_argument("--log_per_updates", action="store_true",
                       help="按实际参数更新次数计算日志间隔")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="手动指定预训练模型路径（优先级最高）")

    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    iter_per_epoch = len(train_loader)
    model, tokenizer = init_model(lm_config)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    for epoch in range(args.epochs):
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
        if os.path.exists(ckp) and args.resume_training:
            checkpoint = torch.load(ckp)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            args.current_step = checkpoint['current_step']
        train_epoch(epoch, wandb)

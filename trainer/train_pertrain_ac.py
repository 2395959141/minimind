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
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings('ignore')

# import os
# os.environ["HF_HUB_OFFLINE"] = "1"


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    # 修改后的预先退火策略（线性预热+余弦退火）
    warmup_ratio = args.warmup_ratio  # 新增预热比例参数
    warmup_steps = int(total_steps * warmup_ratio)
    
    if current_step < warmup_steps:
        # 线性预热阶段
        return lr * (current_step + 1) / warmup_steps
    else:
        # 余弦退火阶段
        decay_steps = total_steps - warmup_steps
        decay_ratio = (current_step - warmup_steps) / decay_steps
        return lr * (math.cos(math.pi * decay_ratio) + 1) * 0.5


def init_pretrain_model(args):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe
    ))

    ckp = os.path.join(args.save_dir, "pretrained_model.pth")  # Define the checkpoint path
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def test_model_on_prompts(model, tokenizer, device, max_seq_len=80):
    prompts = [
        '马克思主义基本原理',
        '人类大脑的主要功能',
        '万有引力定律是',
        '世界上最高的山是',
    ]
    model.eval()
    # 关键修正：兼容 DDP
    if hasattr(model, "module"):
        gen_model = model.module
    else:
        gen_model = model
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
                num_return_sequences=1,
                do_sample=True,
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                top_p=0.85,
                repetition_penalty=1.05,
                temperature=0.85,
                bos_token_id=151643,
                eos_token_id=[
                    tokenizer.convert_tokens_to_ids('<|im_end|>'),
                    tokenizer.convert_tokens_to_ids('<|endoftext|>')
                ] if hasattr(tokenizer, 'convert_tokens_to_ids') else tokenizer.eos_token_id,
            )
            response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"\n[测试] 输入: {prompt}\n[测试] 输出: {response}\n")
    model.train()
    torch.cuda.empty_cache()  # 显式释放显存


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 在每次迭代后添加内存清理
        if step % 500 == 0:
            torch.cuda.empty_cache()
            print("内存清理完成")
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
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

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "训练损失": loss.item() * args.accumulation_steps,
                    "学习率": optimizer.param_groups[-1]['lr'],
                    "每轮训练时间(分钟)": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                }, step=epoch * iter_per_epoch + step)  

        # 新增：每1000 step进行一次模型测试
        if (step + 1) % 5000 == 0 and (not ddp or dist.get_rank() == 0):
            print(f"\n[测试] 当前step: {step+1}，进行模型推理测试：")
            test_model_on_prompts(model, tokenizer, args.device, max_seq_len=80)

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config, args):
    if args.use_qwen25_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    else:
        tokenizer = AutoTokenizer.from_pretrained('./model/')
    model = MiniMindForCausalLM(lm_config).to(args.device)

    # 动态生成检查点路径（指定为 out/pretrain_768.pth）
    checkpoint_path = os.path.join(args.out_dir, "pretrain_768.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        # 检查检查点是否包含 'model_state_dict' 键
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果检查点直接保存了 state_dict，直接加载
            model.load_state_dict(checkpoint)
        print(f"已从检查点恢复模型：{checkpoint_path}")
    else:
        print("未找到检查点，从头开始训练")

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


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="./out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain-minimind-1024")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=16, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="/home/ytllm/.cache/pretrain_data/")
    parser.add_argument("--use_qwen25_tokenizer", type=bool, default=True)
    parser.add_argument("--use_gc", type=bool, default=False)
    parser.add_argument("--warmup_ratio", type=float, default=0.1, 
                       help="预热阶段占总训练步数的比例（0.0~1.0）")
    parser.add_argument("--resume_id", type=str, default=None,help="WandB resume run ID")
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-h{args.hidden_size}-l{args.num_hidden_layers}-bs{args.batch_size}-lr{args.learning_rate}"

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
        try:
            existing_run_id = args.resume_id
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                resume=True,
                id=existing_run_id,
                config=args,
                settings=wandb.Settings(start_method="fork")
            )
            # 设置自定义图表配置
            wandb.define_metric("训练损失", summary="min")
            wandb.define_metric("学习率", summary="last")
        except Exception as e:
            print(f"WandB恢复失败：{e}")
            wandb.init(project=args.wandb_project, name=args.wandb_run_name)  # 创建新运行
            # 设置自定义图表配置
            wandb.define_metric("训练损失", summary="min")
            wandb.define_metric("学习率", summary="last")
    else:
        wandb = None

    model, tokenizer = init_model(lm_config, args)
    ## ! 使用torch.compile进行加速
    # model = torch.compile(model, mode="reduce-overhead")
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    start_epoch = 1  # 根据实际情况调整
    start_step = 38500
    current_step = start_step

    for epoch in range(start_epoch, args.epochs + 1):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)  # 让DDP每轮都shuffle
        train_epoch(epoch - 1, wandb)

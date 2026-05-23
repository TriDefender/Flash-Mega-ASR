# coding=utf-8
from transformers import TrainingArguments

from arguments import parse_args
from checkpointing import MakeCheckpointInferableCallback, find_latest_checkpoint
from dataloader import Qwen3ASRCollator, build_datasets
from modeling import apply_lora, load_qwen3_asr
from trainer import MegaASRTrainer


def build_training_args(args, use_bf16: bool):
    report_to = [] if args.report_to.lower() in ["", "none"] else [args.report_to]

    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=bool(args.pin_memory),
        dataloader_persistent_workers=bool(args.persistent_workers),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        do_eval=bool(args.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=report_to,
        run_name="Mega-ASR-A2S-SFT",
    )


def main():
    args = parse_args()

    model, processor, use_bf16 = load_qwen3_asr(args.model_path)

    if args.padding_side != "auto":
        processor.tokenizer.padding_side = args.padding_side
    print("padding_side =", processor.tokenizer.padding_side)

    model = apply_lora(model, args)

    dataset = build_datasets(args.train_file, args.eval_file)
    collator = Qwen3ASRCollator(processor=processor, sampling_rate=args.sr)
    training_args = build_training_args(args, use_bf16)

    trainer = MegaASRTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        data_collator=collator,
        processing_class=processor,
        callbacks=[MakeCheckpointInferableCallback(args.model_path)],
        processor=processor,
        base_model_path=args.model_path,
        merged_from_lora_path=args.merge_lora_into_base_from.strip(),
        lr_encoder=args.lr_encoder,
        lr_aligner=args.lr_aligner,
        lr_llm=args.lr_llm,
    )

    resume_from = args.resume_from.strip()
    if not resume_from and args.resume:
        resume_from = find_latest_checkpoint(args.output_dir) or ""

    if resume_from:
        print(f"[resume] {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()

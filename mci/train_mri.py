import argparse
import torch
from mri_model import train_new_model, train_smoke_test


def parse_args():
    p = argparse.ArgumentParser(description='Train MRI classifier (VGG / DenseNet)')
    p.add_argument('--model', choices=['vgg', 'densenet'], default='vgg', help='Model architecture')
    p.add_argument('--mode', choices=['fast', 'full', 'smoke'], default='fast', help='Training mode: fast (quick), full (full training), smoke (very short smoke test)')
    p.add_argument('--batches', type=int, default=10, help='Number of batches for smoke test')
    p.add_argument('--no-amp', dest='use_amp', action='store_false', help='Disable automatic mixed precision (AMP) even if CUDA is available')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.mode == 'smoke':
        print('Running smoke test (fast settings) ...')
        elapsed, processed = train_smoke_test(model_type=args.model, batches=args.batches)
        print(f'Smoke test done: processed {processed} samples in {elapsed:.2f}s ({processed/elapsed:.1f} samples/s)')
        raise SystemExit(0)

    fast = args.mode == 'fast'
    print(f"Starting training: model={args.model}, mode={'FAST' if fast else 'FULL'}, use_amp={args.use_amp}")
    clf, acc = train_new_model(model_type=args.model, fast_train=fast, use_amp=args.use_amp)
    print(f"Training finished. Test accuracy: {acc:.4f}")

"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json, sys
import stanza
import spacy_stanza
import numpy as np
import torch as th
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion.test_util import get_weights, denoised_fn_round
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
sys.path.insert(0, 'diffusion_lm/transformers/examples/pytorch/language-modeling')
from custom_trainer import Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree
from infill_util import langevin_fn3, get_score, langevin_fn3_compose, langevin_fn1, langevin_fn4, langevin_fn_tree, langevin_fn_length
from spacy.lang.en import English
import numpy as np
import time

th.set_float32_matmul_precision('high')

def main():
    set_seed(101)
    args = create_argparser().parse_args()
    print(args.diffusion_steps)
    steps = args.diffusion_steps

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], f"training_args.json")
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)
    
    args.diffusion_steps=steps
    args.noise_level = 0.0
    args.sigma_small = True
    args.out_dir = args.out_dir+f"diff_{steps}"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.eval_task_.startswith('control_'):
        args.diffusion_steps = 200  # 500  # DEBUG

    # distributed computing
    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    try:
        model.load_state_dict(th.load(args.model_path))
    except:
        model=th.compile(model)
        model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()

    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])
    model_embs, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                   os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*'*80)
        model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda()
    model3 = get_weights(model_embs, args)
    logger.log('load the partial sequences')
    if args.partial_seq:
        partial_seq = [args.partial_seq]
        partial_seq_idx = ['0']
    elif args.partial_seq_file:
        # implies that we should read from the files
        nlp = English()
        tokenizer_spacy = nlp.tokenizer
        print(f'reading from the file {args.partial_seq_file}', '-*'*20)
        with open(args.partial_seq_file, 'r') as f:
            sent_lst = json.load(f)
        partial_seq = []
        partial_seq_idx = []
        for idx, (key, val) in enumerate(sent_lst.items()):
            # ! Changed to fit multiple padded sections
            # if idx < int(args.start_idx) or idx > int(args.end_idx):
            #     continue
            # partial_seq_ = f"{val['obs1']} " + "PAD " * 10 + f"{val['obs2']}"
            partial_seq_ = val['text']
            word_lst = [x.text for x in tokenizer_spacy(partial_seq_)]
            partial_seq_ = " ".join(word_lst)
            partial_seq.append(partial_seq_)
            partial_seq_idx.append(str(idx))
    else:
        partial_seq = ['A kid friendly venue named Alimentum is located on the riverside .',
                       'Alimentum , situated by the river , is quite child friendly .']
        partial_seq_idx = ['0', '1']
    # else:  generate them by randomly preturbing the inputs data.
    if args.modality in ['synth', 'pos']:
        tokens2id = {v:k for k, v in tokenizer.items()}
        todo_pad_token = tokens2id['END']
        print(f'pad token = {todo_pad_token}')
        encoded_partial_seq = [th.LongTensor([tokens2id[x] for x in seq.split()]) for seq in partial_seq]
        print(encoded_partial_seq[0], len(encoded_partial_seq[0]))
    elif args.modality in ['e2e-tgt', 'roc', 'roc-aug']:
        tokens2id = {v:k for k, v in tokenizer.items()}
        todo_pad_token = -1
        pad_token = tokens2id['PAD']
        encoded_partial_seq = [th.LongTensor([tokens2id.get(x, tokens2id['UNK']) for x in seq.split()]) for seq in partial_seq]
        if args.eval_task_ == 'infill':
            todo_pad_token = tokens2id['PAD']
            print(f'pad token = {todo_pad_token}')
            partial_seq = [(b, a) for (a,b) in zip(partial_seq, partial_seq_idx)]
            pass


    logger.log("sampling...")

    s=time.time()
    sample_dict = {}
    if True:
        for (encoded_seq, control_helper) in zip(encoded_partial_seq, partial_seq) :
            all_images = []
            all_labels = []
            # print(args.num_samples, encoded_seq.shape, 'encoded_seq.shape')
            while len(all_images) * args.batch_size < args.num_samples:
                model_kwargs = {}
                # print(encoded_seq.shape)
                encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size,-1)
                # print(model_embs.weight.device, encoded_seq.device)
                partial_mask_temp = (encoded_seq == todo_pad_token).view(args.batch_size, -1)
                # encoded_seq[encoded_seq == todo_pad_token] = 0
                encoded_seq = encoded_seq.clone().masked_fill_(encoded_seq == todo_pad_token, 3)
                encoded_seq_hidden = model_embs(encoded_seq.cuda())
                seqlen = encoded_seq.size(1)
                if args.model_arch == '1d-unet':
                    encoded_seq_hidden = encoded_seq_hidden.permute(0, 2, 1)
                    partial_mask = partial_mask_temp.unsqueeze(1).expand(-1, args.in_channel, -1)
                    sample_shape = (args.batch_size, args.in_channel, seqlen)
                else:
                    partial_mask = partial_mask_temp.unsqueeze(-1).expand(-1, -1, args.in_channel)
                    sample_shape = (args.batch_size, seqlen, args.in_channel, )
                # print(partial_mask, encoded_seq_hidden.shape)

                label_class_attributes = control_helper
                loop_func_ = diffusion.p_sample_loop_progressive_infill

                for sample in loop_func_(
                        model,
                        sample_shape,
                        encoded_seq_hidden,
                        partial_mask,
                        denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        device=encoded_seq_hidden.device,
                        greedy=False,
                ):
                    final = sample["sample"]
    
                sample = final
        
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                logger.log(f"created {len(all_images) * args.batch_size} samples")
    
            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]
            if args.verbose == 'pipe':
                sample_dict[tuple(label_class_attributes)] = arr
                # print(f'writing to sample_dict, for class {" ".join(label_class_attributes)}')

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'

    e = time.time()
    print(f"!@# {args.diffusion_steps} time elapsed: {e-s} seconds")

    dist.barrier()
    logger.log("sampling complete")

    def decode_helper(args, sample_dict, diff_model=None):
        result_dict = {}
        if not diffusion.training_mode.startswith('e2e'):
            logger.log('decode by rounding. ')
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])

        for k, v in sample_dict.items():
            # print(k,v)
            arr = v
            if diffusion.training_mode.startswith('e2e'):
                word_lst_e2e = []
                # print('decoding for e2e', )
                x_t = th.tensor(arr).cuda()
                # print(x_t.shape)
                if args.model_arch == 'conv-unet':
                    reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
                else:
                    reshaped_x_t = x_t
                logits = diff_model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
                cands = th.topk(logits, k=1, dim=-1)
                tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
                for seq in cands.indices:
                    tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                    word_lst_e2e.append(tokens)
                word_lst = word_lst_e2e
            else:
                word_lst = rounding_func(args.experiment, arr, model, tokenizer)
            result_dict[k] = word_lst
        
        return result_dict

    if args.verbose == 'pipe':
        print(f'sampled for {len(sample_dict)} control tasks')
        out_path_pipe = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.notes}.txt")
        result_dict = decode_helper(args, sample_dict, diff_model=model)
        with open(out_path_pipe, 'w+') as f:
            for k, result in result_dict.items():
                for sen in result:
                    f.write(sen.replace('\n', '\\n')+'\n')
        print(f'written the decoded output to {out_path_pipe}')
    args.out_path2 = out_path_pipe
    return args

def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=8, batch_size=1, model_path="",
        out_dir="diffusion_lm/improved-diffusion/out_gen/",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
        start_idx=0, end_idx=0,base=None,diffusion_steps=2000
    )
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser = add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = main()
    if args.verbose != 'pipe':
        eval(args)

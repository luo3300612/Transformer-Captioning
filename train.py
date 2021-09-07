import pdb
import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer
from evaluation.cider import Cider
import json
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import time
from data import build_image_field
from models import model_factory
from line_profiler import LineProfiler
from contiguous_params import ContiguousParams

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
stoi_for_cider = {}


def evaluate_loss(model, dataloader, loss_fn, text_field, test=False):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

                if test:
                    break

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field, test=False):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()
            if test:
                break

    gts = evaluation.PTBTokenizer.tokenize(gts)  # 这里做没啥问题，因为多轮Tokenize在验证/测试集上没影响
    gen = evaluation.PTBTokenizer.tokenize(gen)  #
    print('examples:')
    print('gen:', gen['0_0'])
    print('gt:', gts['0_0'])
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field, test=False):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            out = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

            loss = loss.mean()
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            parameters.assert_buffer_is_valid()
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            # scheduler.step()

            if test:
                break

            # if it == 99:
            #     break

    loss = running_loss / len(dataloader)
    return loss


def my_decode(word_idxs, eos_idx, join_words=True):
    captions = {}
    for i, wis in enumerate(word_idxs):
        caption = []
        for wi in wis:
            word = int(wi)
            if word == eos_idx:
                break
            caption.append(word)
        # pdb.set_trace()
        if join_words:
            caption = ' '.join(caption)
        captions[i] = [caption]
    return captions


def encode_caps_gt(caps_gts):
    encoded_caps_gt = {}
    for i, caps_gt in enumerate(caps_gts):
        refs = []
        for sentence in caps_gt:
            words = sentence.split(' ')
            try:
                indexes = [stoi_for_cider[word] for word in words]
            except KeyError:
                print('raw sentence')
                print(sentence)
                raise
            refs.append(indexes)
        encoded_caps_gt[i] = refs
    return encoded_caps_gt


def train_scst(model, dataloader, optim, cider, text_field, test=False):
    # Training with self-critical
    # tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            replicated_caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            # Rewards
            my_caps_gen = my_decode(outs.view(-1, seq_len), text_field.vocab.stoi['<eos>'], join_words=False)
            my_caps_gt = encode_caps_gt(replicated_caps_gt)

            reward = cider.compute_score(my_caps_gt, my_caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            parameters.assert_buffer_is_valid()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

            if test:
                break

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    torch.cuda.synchronize()
    return loss, reward, reward_baseline


def encode_corpus(corpus, stoi):
    encoded_corpus = {}
    fix_stoi = {}
    stoi_copy = {}  # use copy since stoi is NOT A NORMAL DICT!!!
    for k, v in stoi.items():
        stoi_copy[k] = v

    for key, value in tqdm(corpus.items(), desc='encode corpus'):
        words = value[0].split(' ')
        encoded_words = []
        for word in words:
            index = stoi_copy.get(word, 0)
            if index == 0:
                index = fix_stoi.get(word, None)
                if index is None:
                    fix_stoi[word] = len(stoi_copy) + len(fix_stoi)
                    index = fix_stoi[word]
            encoded_words.append(index)
        encoded_corpus[key] = [encoded_words]

    for k, v in stoi_copy.items():
        stoi_for_cider[k] = v
    for k, v in fix_stoi.items():
        stoi_for_cider[k] = v
    return encoded_corpus


def make_corpus(ref_train):
    res = {}
    for i, item in enumerate(ref_train):
        res[i] = [item]
    return res


def build_cider_train(ref_caps_train, args):
    # corpus = PTBTokenizer.tokenize(ref_caps_train)
    corpus = make_corpus(ref_caps_train)
    # cider_train = Cider(corpus)
    encoded_corpus = encode_corpus(corpus, text_field.vocab.stoi)
    if not os.path.isfile('.vocab_cache/cider.pkl' % args.exp_name):
        cider_train = Cider(encoded_corpus, get_cache=True)
        if not os.path.exists('.vocab_cache'):
            os.mkdir('.vocab_cache')
        pickle.dump(cider_train.gts_cache, open('.vocab_cache/cider.pkl' % args.exp_name, 'wb'))
    else:
        print('loading origin_cider cache')
        cider_train = Cider(encoded_corpus)
        cider_train.gts_cache = pickle.load(open('.vocab_cache/cider.pkl' % args.exp_name, 'rb'))
    return cider_train


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Transformer captioning')
    parser.add_argument('--exp_name', type=str, default='anonymous_run')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--grid_on', action='store_true', default=False)
    parser.add_argument('--rl_batch_size', type=int, default=50)
    parser.add_argument('--rl_learning_rate', type=float, default=5e-6)
    parser.add_argument('--max_detections', type=int, default=50)
    parser.add_argument('--dim_feats', type=int, default=2048)
    parser.add_argument('--image_field', type=str, default="ImageDetectionsField")
    parser.add_argument('--model', type=str, default="transformer")
    parser.add_argument('--rl_at', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Transformer Training')

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    start = time.time()
    image_field = build_image_field(args)

    print('image field time')
    print(time.time() - start)
    start = time.time()
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='split',
                           remove_punctuation=True, nopoints=False)
    print('text field time')
    print(time.time() - start)
    start = time.time()

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    print('dataset time')
    print(time.time() - start)
    start = time.time()
    train_dataset, val_dataset, test_dataset = dataset.splits
    print('split time')
    print(time.time() - start)
    start = time.time()

    if not os.path.isfile('.vocab_cache/vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        if not os.path.exists('.vocab_cache'):
            os.mkdir('.vocab_cache')
        pickle.dump(text_field.vocab, open('.vocab_cache/vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('.vocab_cache/vocab_%s.pkl' % args.exp_name, 'rb'))

    print('build vocab time')
    print(time.time() - start)
    start = time.time()
    # Model and dataloaders
    Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention = model_factory(args)
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention,
                                 d_in=args.dim_feats,
                                 d_k=args.d_k,
                                 d_v=args.d_v,
                                 h=args.head
                                 )
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'],
                                      d_k=args.d_k,
                                      d_v=args.d_v,
                                      h=args.head
                                      )
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    parameters = ContiguousParams(model.parameters())

    print('build model time')
    print(time.time() - start)
    start = time.time()
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = None
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    print('prepare dataset time')
    print(time.time() - start)


    def lambda_lr(s):
        base_lr = 0.0001
        print("s:", s)
        if s <= 3:
            lr = base_lr * s / 4
        elif s <= 10:
            lr = base_lr
        elif s <= 12:
            lr = base_lr * 0.2
        else:
            lr = base_lr * 0.2 * 0.2
        # s += 1
        return lr


    # Initial conditions
    optim = Adam(parameters.contiguous(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])  # ,reduction='none')
    use_rl = False
    best_cider = .0
    best_test_cider = .0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best origin_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.rl_batch_size // 5, shuffle=True,
                                       num_workers=args.workers)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.rl_batch_size // 5)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.rl_batch_size // 5)

    if args.test:
        # test each method for 1 iteration
        e = 0
        print('test start')
        train_xe(model, dataloader_train, optim, text_field, test=True)
        evaluate_loss(model, dataloader_val, loss_fn, text_field, test=True)
        evaluate_metrics(model, dict_dataloader_test, text_field, test=True)

        image_field.f.close()
        del image_field.f
        # corpus = PTBTokenizer.tokenize(ref_caps_train)
        corpus = make_corpus(ref_caps_train)
        encoded_corpus = encode_corpus(corpus, text_field.vocab.stoi)
        cider_train = build_cider_train(ref_caps_train, args)
        del dataloader_train
        train_scst(model, dict_dataloader_train, optim, cider_train, text_field, test=True)
        print('test done')
        exit(0)

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        if hasattr(image_field, 'f'):
            image_field.f.close()
            del image_field.f

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            if cider_train is None:
                cider_train = build_cider_train(ref_caps_train, args)
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train,
                                                             text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", scores)
        test_cider = scores['CIDEr']
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True

        switch_to_rl = False
        exit_train = False
        if e == args.rl_at:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                print("Switching to RL")
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load('saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            parameters = ContiguousParams(model.parameters())
            optim = Adam(parameters.contiguous(), lr=args.rl_learning_rate)

            print('Resuming from epoch %d, validation loss %f, and best origin_cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s_last.pth' % args.exp_name)

        if best:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)

        if best_test:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best_test.pth' % args.exp_name)

        if switch_to_rl:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_xe_res.pth' % args.exp_name)

        if exit_train:
            writer.close()
            break

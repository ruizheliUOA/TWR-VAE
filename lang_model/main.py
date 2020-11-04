import argparse
from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import random
import math
import os
import time
from BatchIter import BatchIter
import numpy as np
from Corpus import Corpus
from encoder import Encoder
from decoder import Decoder

parser = argparse.ArgumentParser(description='TWR-VAE for PTB/Yelp/Yahoo')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--embedding_size', type=int, default=512,
                    help='embedding size for training (default: 512)')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='hidden size for training (default: 256)')
parser.add_argument('--zdim',  type=int, default=32,
                    help='the z size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--layers', type=int, default=1,
                    help='number of layers of rnns in encoder and decoder')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout values')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--no_cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('-s', '--save', action='store_true', default=True,
                    help='save model every epoch')
parser.add_argument('-l', '--load', action='store_true',
                    help='load model at the begining')
parser.add_argument('-dt','--dataset', type=str, default="ptb",
                    help='Dataset name')
parser.add_argument('-mw','--min_word_count',type=int, default=1,
                    help='minimum word count')
parser.add_argument('-st','--setting',type=str, default='standard',
                    help='standard setting or inputless setting')
parser.add_argument('-rnn','--rnn_type', type=str, default="lstm",
                    help='RNN types (rnn, lstm and gru)')
parser.add_argument('-par','--partial', action='store_true', 
                    help='partially optimise KL')
parser.add_argument('-party','--partial_type', type=str, default='last75',
                    help='partial type: last1 last25 last50 last75')
parser.add_argument('--z_type', type=str, default='normal',
                    help='z mode for decoder: normal, mean, sum')
parser.add_argument('--model_dir', type=str, default='',
                    help='model storing path')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')
args = parser.parse_args()

print(args)

base_path = "./lang_model"
print("base_path=", base_path)

if args.dataset == "ptb":
    Train = Corpus(base_path+'/dataset/ptb/ptb_train.txt', min_word_count=args.min_word_count)
    Valid = Corpus(base_path+'/dataset/ptb/ptb_val.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/dataset/ptb/ptb_test.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)
elif args.dataset == "yelp":
    Train = Corpus(base_path+'/dataset/yelp/yelp.train.txt', min_word_count=args.min_word_count)
    Valid = Corpus(base_path+'/dataset/yelp/yelp.valid.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/dataset/yelp/yelp.test.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)
elif args.dataset == "yahoo":
    Train = Corpus(base_path+'/dataset/yahoo/yahoo_train.txt', min_word_count=args.min_word_count)
    Valid = Corpus(base_path+'/dataset/yahoo/yahoo_val.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/dataset/yahoo/yahoo_test.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)

if args.load:
    model_dir = args.model_dir
    recon_dir = base_path+'/'+args.dataset+'_recon_save/'

else:

    model_dir = base_path+'/'+args.dataset+'_model_save/'
    recon_dir = base_path+'/'+args.dataset+'_recon_save/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)


voca_dim = Train.voca_size

print(f"voca_dim={voca_dim}")


emb_dim = args.embedding_size
hid_dim = args.hidden_size
batch_size = args.batch_size
if args.setting == 'standard':
    teacher_force = 1
elif args.setting == 'inputless':
    teacher_force = 0

SEED = 999
lr = args.lr


random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available()
                      and not args.no_cuda else 'cpu')


dataloader_train = BatchIter(Train, batch_size)
dataloader_valid = BatchIter(Valid, batch_size)
dataloader_test = BatchIter(Test, batch_size)


encoder = Encoder(voca_dim, emb_dim, hid_dim, args.zdim, args.layers, args.dropout, 
                  rnn_type=args.rnn_type, 
                  partial=args.partial_type, 
                  z_mode=args.z_type, 
                  partial_lag=args.partial).to(device)
decoder = Decoder(voca_dim, emb_dim, hid_dim, args.zdim, args.layers, args.dropout,
                  teacher_force=teacher_force, 
                  rnn_type=args.rnn_type, 
                  z_mode=args.z_type, 
                  setting=args.setting, 
                  device=device).to(device)
opt = optim.Adam(list(encoder.parameters()) +
                 list(decoder.parameters()), lr=lr, eps=1e-6, weight_decay=1e-5)

print(encoder)
print(decoder)

def sentence_acc(prod, target):
    target = target[1:]
    mask = target == 0
    prod = prod.argmax(dim=2)
    prod[mask] = -1
    correct = torch.eq(prod, target).to(dtype=torch.float).sum()
    return correct.item()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(corpus, ep):
    
    print("--------------------------")
    start_time = time.time()
    encoder.train()
    decoder.train()
    total = 0
    recon_loss_total = 0
    kl_loss_total = 0
    correct_total = 0
    words_total = 0
    batch_total = 0
    
    
    for i, sen in enumerate(corpus):
        
        # sen: [len_sen, batch]
        batch_size = sen.shape[1]
        opt.zero_grad()
        total += sen.shape[1]
        sen = sen.to(device)
        z, mu, logvar, sen_len, _ = encoder(sen)


        
       
        prod = decoder(z, sen)
        kl_loss = encoder.loss(mu, logvar)
        recon_loss = decoder.loss(prod, sen, sen_len)
        
        
        ((kl_loss+recon_loss)*1).backward()
        opt.step()
        

        recon_loss_total = recon_loss_total + recon_loss.item()
        kl_loss_total = kl_loss_total + kl_loss.item()
        correct = sentence_acc(prod, sen)
        words = sen_len.sum().item()
        correct_total = correct_total + correct
        words_total = words_total + words
        batch_total += batch_size
        
        
        

    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(
        f"Time: {epoch_mins}m {epoch_secs}s| Train {ep}: recon_loss={(recon_loss_total/(batch_total)):.04f}, kl_loss={(kl_loss_total/(batch_total)):.04f}, nll_loss={((recon_loss_total+kl_loss_total)/(batch_total)):.04f}, nll_loss_perword={((recon_loss_total+kl_loss_total)/words_total):.04f}, ppl={(math.exp((recon_loss_total+kl_loss_total)/words_total)):.04f}, acc={(correct_total/words_total):.04f}")
    return recon_loss_total/(batch_total), kl_loss_total/(batch_total),  correct_total/words_total, (recon_loss_total+kl_loss_total)/(batch_total), math.exp((recon_loss_total+kl_loss_total)/words_total)

# =================================


def get_sentence(batch):
    sens = []
    for b in range(batch.shape[1]):
        sen = [Train.id_word[batch[i, b].item()]
               for i in range(batch.shape[0])]
        sens.append(" ".join(sen))
    return sens



def reconstruction(corpus, ep):
    encoder.eval()
    decoder.eval()
    out_org = []
    out_recon_mu = []
    recon_loss_total = 0
    kl_loss_total = 0
    words_total = 0
    batch_total = 0
    mi_total = 0

  
    
    
    for i, sen in enumerate(corpus):
        b_size = sen.shape[1]
        out_org += get_sentence(sen[1:])
        sen = sen.to(device)
        
        with torch.no_grad():
            z, mu, logvar, sen_len, mi_per_batch = encoder(sen)
            
            
            recon_mu = decoder(z, sen)
            kl_loss = encoder.loss(mu, logvar)
            recon_loss = decoder.loss(recon_mu, sen, sen_len)
            
            sens_mu = recon_mu.argmax(dim=2)
            out_recon_mu += get_sentence(sens_mu.to("cpu"))

            recon_loss_total = recon_loss_total + recon_loss.item()
            kl_loss_total = kl_loss_total + kl_loss.item()
            
            words = sen_len.sum().item()
            
            words_total = words_total + words
            batch_total += b_size
            mi_total += mi_per_batch * b_size
            
    print(f"Eval: recon_loss:{(recon_loss_total/(batch_total)):.04f}, kl_loss:{(kl_loss_total/(batch_total)):.04f}, nll_loss:{((recon_loss_total+kl_loss_total)/(batch_total)):.04f}, nll_loss_perword={((recon_loss_total+kl_loss_total)/words_total):.04f}, ppl:{(math.exp((recon_loss_total+kl_loss_total)/words_total)):.04f}, mi:{(mi_total/batch_total):.04f}")



    
    text = []
    for i in range(len(out_recon_mu)):
        text.append("origion: " + out_org[i])
        text.append("reco_mu: " + out_recon_mu[i])
        text.append("\n")
    with open(recon_dir+f"TWRvae_outcome_{ep}.txt", "w") as f:
        f.write("\n".join(text))

    
            

    return math.exp((recon_loss_total+kl_loss_total)/words_total)



ep = 0
# ============== run ==============
if args.load:
    state = torch.load(model_dir+'TWRvae.tch')
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    state2 = torch.load(model_dir+'TWRvae.tchopt')
    ep = state2["ep"]+1
    opt.load_state_dict(state2["opt"])
    eval_ppl = reconstruction(dataloader_valid, ep)
    test_ppl = reconstruction(dataloader_test, ep)
else:
    history = []
    Best_ppl = 1e5
    for ep in range(ep, args.epochs):
        recon_loss, var_loss, acc, nll_loss, ppl = train(dataloader_train, ep)
        history.append(f"{ep}\t{recon_loss}\t{var_loss}\t{acc}\t{nll_loss}\t{ppl}")
        with open(model_dir+'TWRvae_loss.txt', 'w') as f:
            f.write("\n".join(history))
        if ep % 10 == 0:
            eval_ppl = reconstruction(dataloader_valid, ep)
            test_ppl = reconstruction(dataloader_test, ep)
            
        if args.save and eval_ppl < Best_ppl:
            Best_ppl = eval_ppl
            
            state = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }
            torch.save(state, model_dir + 'TWRvae.tch')
            state2 = {
                "opt": opt.state_dict(),
                "ep": ep
            }
            torch.save(state2, model_dir + 'TWRvae.tchopt')
    

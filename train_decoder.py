import argparse

import clip
import torch
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig, BertTokenizer
import os
from dataloaders.coco_full_loader import get_loader
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from torch.optim import AdamW
from tqdm import tqdm


def train_decoder(bert_model, train_loader, eval_loader, optimizer):
    early_stop = 0
    num_batch = len(iter(train_loader))
    print(f"Starting training for max {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        acc_loss = 0
        print('Training : epoch {}'.format(epoch))
        for i, batch in enumerate(tqdm(train_loader)):
            # if i==1:break
            input_ids, attention_mask, label_ids, clip_embeds = batch
            clip_extended_embed = clip_embeds.repeat(1, 2).type(torch.FloatTensor)

            N, seq_length = input_ids.shape
            position_ids = torch.arange(0, seq_length).expand(N, seq_length)
            bert_model.train()
            out = bert_model(input_ids=input_ids.to(device),
                             position_ids=position_ids.to(device),
                             attention_mask=attention_mask.to(device),
                             encoder_hidden_states=clip_extended_embed.unsqueeze(1).to(device),
                             labels=label_ids.to(device))

            out.loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()
            acc_loss += out.loss.detach().item()

        validation_loss = eval_decoder(bert_model, eval_loader)
        print('validation loss in this epoch: ', validation_loss)
        state = {'net': bert_model.state_dict(),
                 'epoch': epoch,
                 'validation loss': validation_loss}

        if epoch == 0:
            best_val_loss = validation_loss
            torch.save(state, args.saved_model_path + 'model_dump.pt')
        else:
            if validation_loss < best_val_loss:
                early_stop = 0
                best_val_loss = validation_loss
                torch.save(state, args.saved_model_path + 'model_2.pt')
            else:
                early_stop += 1

        print('Average loss on {} training batches in this epoch:{}\n'.format(num_batch, acc_loss / num_batch))

        if early_stop >= 3:
            print(f"No improvements on val data for {early_stop} iterations. Breaking now")
            break
        
    return acc_loss


def eval_decoder(bert_model, eval_loader):
    num_batch = len(iter(eval_loader))
    print('evaluating loss on validation data ...')
    acc_loss = 0
    bert_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader)):
            input_ids, attention_mask, label_ids, clip_embeds = batch
            clip_extended_embed = clip_embeds.repeat(1, 2).type(torch.FloatTensor)

            N, seq_length = input_ids.shape
            position_ids = torch.arange(0, seq_length).expand(N, seq_length)
            out = bert_model(input_ids=input_ids.to(device),
                             position_ids=position_ids.to(device),
                             attention_mask=attention_mask.to(device),
                             encoder_hidden_states=clip_extended_embed.unsqueeze(1).to(device),
                             labels=label_ids.to(device))
            acc_loss += out.loss.detach().item()
    print('Average loss on {} validation batches={}\n'.format(num_batch, acc_loss / num_batch))
    return acc_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=1, help="End epoch")  # trained with 25 epochs
    parser.add_argument('--trained_path', type=str, default='./trained_models/COCO/')
    parser.add_argument('--bert_model', type=str, default='google/bert_for_seq_generation_L-24_bbc_encoder')
    parser.add_argument('--clip_vision', type=str, default='ViT-B/32')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.saved_model_path = args.trained_path + 'ViT-B32'

    if not os.path.exists(args.saved_model_path):
        os.makedirs(args.saved_model_path)

    # initialize tokenizers for clip and bert, these two use different tokenizers
    if args.bert_model == "google/bert_for_seq_generation_L-24_bbc_encoder":
        berttokenizer = BertGenerationTokenizer.from_pretrained(args.bert_model)
    else:
        berttokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # clip_model = torch.jit.load(os.path.join('./trained_models', "{}.pt".format('ViT-B32'))).to(device).eval()
    clip_model, _ = clip.load(args.clip_vision)
    # loader to get preprocessed and encoded (image, caption) from COCO dataset
    train_loader = get_loader(train=True, clip_backbone=args.clip_vision, clip_model=clip_model,
                              berttokenizer=berttokenizer)
    eval_loader = get_loader(train=False, clip_backbone=args.clip_vision, clip_model=clip_model,
                             berttokenizer=berttokenizer)

    # load clip pretrained image encoder

    bert_config = BertGenerationConfig.from_pretrained(args.bert_model)
    bert_config.is_decoder = True
    bert_config.add_cross_attention = True
    bert_model = BertGenerationDecoder.from_pretrained(args.bert_model,
                                                       config=bert_config).to(device).train()

    optimizer = AdamW(bert_model.parameters(), lr=args.lr)

    loss = train_decoder(bert_model, train_loader, eval_loader, optimizer)
    print('final training loss={}'.format(loss))

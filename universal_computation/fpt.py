import json
import torch
import numpy as np
import torch.nn as nn
import wandb

class FPT(nn.Module):

    def __init__(
            self,
            input_dim,
            output_dim,
            model_name='gpt2',
            pretrained=False,
            return_last_only=True,
            use_embeddings_for_in=False,
            in_layer_sizes=None,
            out_layer_sizes=None,
            task=None,
            freeze_trans=True,
            freeze_in=False,
            freeze_pos=True,
            freeze_ln=True,
            freeze_attn=True,
            freeze_ff=True,
            freeze_out=True,
            freeze_other=True,
            optimized=False,
            optimized_type=None,
            dropout=0.1,
            orth_gain=1.41,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.return_last_only = return_last_only
        self.use_embeddings_for_in = use_embeddings_for_in

        self.in_layer_sizes = [] if in_layer_sizes is None else in_layer_sizes
        self.out_layer_sizes = [] if out_layer_sizes is None else out_layer_sizes
        self.dropout = dropout

        if 'gpt' in model_name:
            assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

            from transformers import GPT2Model

            pretrained_transformer = GPT2Model.from_pretrained(model_name)
            if pretrained:
                self.sequence_model = pretrained_transformer
            else:
                self.sequence_model = GPT2Model(pretrained_transformer.config)

            if model_name == 'gpt2':
                embedding_size = 768
            elif model_name == 'gpt2-medium':
                embedding_size = 1024
            elif model_name == 'gpt2-large':
                embedding_size = 1280
            elif model_name == 'gpt2-xl':
                embedding_size = 1600

        elif model_name == 'vit':

            import timm

            self.sequence_model = timm.create_model(
                'vit_base_patch16_224', pretrained=pretrained, drop_rate=dropout, attn_drop_rate=dropout,
            )
            embedding_size = 768

            self.vit_pos_embed = nn.Parameter(torch.zeros(1, 1024, embedding_size))
            if freeze_pos:
                self.vit_pos_embed.requires_grad = True

        elif model_name == 'lstm':

            from universal_computation.models.lstm import LNLSTM

            num_layers, embedding_size = 3, 768

            self.sequence_model = LNLSTM(
                input_size=embedding_size,
                hidden_size=embedding_size,
                num_layers=num_layers,
                batch_first=True,
                residual=False,
                dropout=dropout,
                bidirectional=0,
            )

            # optionally:
            # self.lstm_pos_embed = nn.Parameter(torch.zeros(1, 1024, embedding_size))
            # if freeze_pos:
            #     self.lstm_pos_embed.requires_grad = False

        else:
            raise NotImplementedError('model_name not implemented')

        if use_embeddings_for_in:
            self.in_net = nn.Embedding(input_dim, embedding_size)
        else:
            in_layers = []
            last_output_size = input_dim
            for size in self.in_layer_sizes:
                layer = nn.Linear(last_output_size, size)
                if orth_gain is not None:
                    torch.nn.init.orthogonal_(layer.weight, gain=orth_gain)
                layer.bias.data.zero_()

                in_layers.append(layer)
                in_layers.append(nn.ReLU())
                in_layers.append(nn.Dropout(dropout))
                last_output_size = size

            final_linear = nn.Linear(last_output_size, embedding_size)
            if orth_gain is not None:
                torch.nn.init.orthogonal_(final_linear.weight, gain=orth_gain)
            final_linear.bias.data.zero_()

            in_layers.append(final_linear)
            in_layers.append(nn.Dropout(dropout))

            self.in_net = nn.Sequential(*in_layers)

        out_layers = []
        last_output_size = embedding_size
        for size in self.out_layer_sizes:
            out_layers.append(nn.Linear(last_output_size, size))
            out_layers.append(nn.ReLU())
            out_layers.append(nn.Dropout(dropout))
            last_output_size = size
        out_layers.append(nn.Linear(last_output_size, output_dim))
        self.out_net = nn.Sequential(*out_layers)

        if freeze_trans and not optimized:
            for name, p in self.sequence_model.named_parameters():
                name = name.lower()
                if 'ln' in name or 'norm' in name:
                    p.requires_grad = not freeze_ln
                elif 'wpe' in name or 'position_embeddings' in name or 'pos_drop' in name:
                    p.requires_grad = not freeze_pos
                elif 'mlp' in name:
                    p.requires_grad = not freeze_ff
                elif 'attn' in name:
                    p.requires_grad = not freeze_attn
                else:
                    p.requires_grad = not freeze_other
        elif optimized and optimized_type=='variance':
            #try:
                path = "/root/universal-computation/"+task+"-data.json"    
                with open(path) as file:
                    grad_dict = json.load(file)
                var_dict = dict()
                for key in grad_dict.keys():
                    var_dict[key] = torch.var(torch.tensor(grad_dict[key]))
                sorted_var_dict = dict(sorted(var_dict.items(), key=lambda x: x[1]))
                value_list = list()
                key_list = list()
                for key, value in sorted_var_dict.items():
                    value_list.append(value)
                    key_list.append(key)
                value_list = np.array(np.delete(value_list,0))
                key_list = key_list[1:]
                counter=0
                cumulative_list = np.array(np.cumsum(value_list)/np.sum(value_list))
                for value in cumulative_list:
                    if value > 0.01:
                        break
                    else:
                        counter+=1
                value=counter-1
                total_parameters = 0
                trainable_parameters = 0
                for name, p in self.sequence_model.named_parameters():
                    name = name.lower()
                    total_parameters += torch.numel(p)
                    for key in key_list[value:]: 
                        if name in key:
                            p.requires_grad = True
                            trainable_parameters += torch.numel(p)
                        else:
                            p.requires_grad = False
                print('=' * 57)
                print(trainable_parameters)
                print(total_parameters)
                print(f'Trainable Parameter Percentage: {trainable_parameters/total_parameters}')
                print('=' * 57)
            #except:
                #raise NotImplementedError('json file not found')
        elif optimized and optimized_type=='snr':
            path = "/root/universal-computation/"+task+"-data.json"
            with open(path) as file:
                grad_dict = json.load(file)
            snr_dict = dict()
            for key in grad_dict.keys():
                var = torch.var(torch.tensor(grad_dict[key]))
                mean = torch.mean(torch.tensor(grad_dict[key]))
                snr_dict[key] = (mean ** 2)/var
            sorted_snr_dict = dict(sorted(snr_dict.items(), key=lambda x: x[1]))
            value_list = list()
            key_list = list()
            for key, value in sorted_snr_dict.items():
                value_list.append(value)
                key_list.append(key)
            value_list = np.array(np.delete(value_list,0))
            key_list = key_list[1:]
            counter=0
            cumulative_list = np.array(np.cumsum(value_list)/np.sum(value_list))
            for value in cumulative_list:
                if value > 0.01:
                    break
                else:
                    counter+=1
            value=counter-1
            total_parameters = 0
            trainable_parameters = 0
            for name, p in self.sequence_model.named_parameters():
                name = name.lower()
                total_parameters += torch.numel(p)
                for key in key_list[value:]: 
                    if name in key:
                        p.requires_grad = True
                        trainable_parameters += torch.numel(p)
                    else:
                        p.requires_grad = False
            print('=' * 57)
            print(trainable_parameters)
            print(total_parameters)
            print(f'Trainable Parameter Percentage: {trainable_parameters/total_parameters}')
            print('=' * 57)           
        
        if freeze_in:
            for p in self.in_net.parameters():
                p.requires_grad = True
        if freeze_out:
            for p in self.out_net.parameters():
                p.requires_grad = True
    
    def forward(self, x):
        # reshape x (batch_size, seq_len, dim) into patches (batch_size, seq_len*num_patches, patch_dim)
        orig_dim = x.shape[-1]
        if orig_dim != self.input_dim and not self.use_embeddings_for_in:
            if orig_dim % self.input_dim != 0:
                raise ValueError('dimension of x must be divisible by patch size')
            ratio = orig_dim // self.input_dim
            x = x.reshape(x.shape[0], x.shape[1] * ratio, self.input_dim)
        else:
            ratio = 1
        x = self.in_net(x)

        # ignore input layer that comes with model and use our own embeddings
        if self.model_name == 'vit':
            x = x + self.vit_pos_embed[:, :x.shape[1]]
            x = self.sequence_model.pos_drop(x)
            for blk in self.sequence_model.blocks:
                x = blk(x)
            x = self.sequence_model.norm(x)
        elif self.model_name == 'lstm':
            # x = x + self.lstm_pos_embed[:, :x.shape[1]]
            x, *_ = self.sequence_model(x)
        else:
            transformer_outputs = self.sequence_model(
                inputs_embeds=x,
                return_dict=True,
            )
            x = transformer_outputs.last_hidden_state

        # take final hidden state of tokens corresponding to last patch
        if self.return_last_only:
            x = x[:,-ratio:]

        # single linear layer applied to last hidden state
        x = self.out_net(x)

        # if we did patch resizing above, return in the original shape (batch_size, seq_len, dim)
        if self.return_last_only and ratio > 1:
            x = x.reshape(x.shape[0], x.shape[1] // ratio, ratio * self.output_dim)

        return x
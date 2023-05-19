import universal_computation.experiment as experiment   

if __name__ == '__main__':

    experiment_name = 'fpt'

    frozen_params = dict(
        task='cifar10',
        n=1000,                # ignored if not a bit task
        num_patterns=5,        # ignored if not a bit task
        patch_size=4,        
        # patch_size=4,

        model_name='gpt2',
        pretrained=True,       # if vit this is forced to true, if lstm this is forced to false

        freeze_trans=True,     # if False, we don't check arguments other than in and out
        freeze_in=False,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
        freeze_out=False,

        training_type='frozen',

        in_layer_sizes=None,   # not in paper, but can specify layer sizes for an MLP,
        out_layer_sizes=None,  # ex. [32, 32] creates a 2-layer MLP with dimension 32

        learning_rate=1e-3,
        batch_size=4,
        dropout=0.1,
        orth_gain=1.41, 
                      # orthogonal initialization of input layer
    )

    finetuned_params = dict(
        task='cifar10',
        n=1000,                # ignored if not a bit task
        num_patterns=5,        # ignored if not a bit task
        patch_size=4,       
        # patch_size=4,

        model_name='gpt2',
        pretrained=True,       # if vit this is forced to true, if lstm this is forced to false

        freeze_trans=True,     # if False, we don't check arguments other than in and out
        freeze_in=False,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
        freeze_out=False,

        training_type='finetune',

        in_layer_sizes=None,   # not in paper, but can specify layer sizes for an MLP,
        out_layer_sizes=None,  # ex. [32, 32] creates a 2-layer MLP with dimension 32

        learning_rate=1e-3,
        batch_size=4,
        dropout=0.1,
        orth_gain=1.41, 
                      # orthogonal initialization of input layer
    )

    optimized_params = dict(
        task='cifar10',
        n=1000,                # ignored if not a bit task
        num_patterns=5,        # ignored if not a bit task
        patch_size=4,          
        # patch_size=4,

        model_name='gpt2',
        pretrained=True,       # if vit this is forced to true, if lstm this is forced to false

        freeze_trans=True,     # if False, we don't check arguments other than in and out
        freeze_in=False,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
        freeze_out=False,

        training_type='optimized',

        in_layer_sizes=None,   # not in paper, but can specify layer sizes for an MLP,
        out_layer_sizes=None,  # ex. [32, 32] creates a 2-layer MLP with dimension 32

        learning_rate=1e-3,
        batch_size=4,
        dropout=0.1,
        orth_gain=1.41, 
                      # orthogonal initialization of input layer
    )

    snr_params = dict(
        task='cifar10',
        n=1000,                # ignored if not a bit task
        num_patterns=5,        # ignored if not a bit task
        patch_size=4,          
        # patch_size=4,

        model_name='gpt2',
        pretrained=True,       # if vit this is forced to true, if lstm this is forced to false

        freeze_trans=True,     # if False, we don't check arguments other than in and out
        freeze_in=False,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
        freeze_out=False,

        training_type='snr',

        in_layer_sizes=None,   # not in paper, but can specify layer sizes for an MLP,
        out_layer_sizes=None,  # ex. [32, 32] creates a 2-layer MLP with dimension 32

        learning_rate=1e-3,
        batch_size=4,
        dropout=0.1,
        orth_gain=1.41, 
                      # orthogonal initialization of input layer
    )

    experiment.run_experiment(experiment_name, frozen_params)
    experiment.run_experiment(experiment_name, finetuned_params)
    experiment.run_experiment(experiment_name, optimized_params)
    experiment.run_experiment(experiment_name, snr_params)
    
import models.spike as spike

def get_model_params(config, num_coord_joints):
    """Get the parameters for the model from the config."""
    return {
        'radius': config.get('radius'),
        'nsamples': config.get('nsamples'),
        'spatial_stride': config.get('spatial_stride'),
        'temporal_kernel_size': config.get('temporal_kernel_size'),
        'temporal_stride': config.get('temporal_stride'),
        'emb_relu': config.get('emb_relu'),
        'dim': config.get('dim'),
        'depth': config.get('depth'),
        'heads': config.get('heads'),
        'dim_head': config.get('dim_head'),
        'mlp_dim': config.get('mlp_dim'),
        'num_coord_joints': num_coord_joints,
        'dropout1': config.get('dropout1'),
        'dropout2': config.get('dropout2')
    }

def create_model(config, num_coord_joints):
    """Create a model based on the config and return it."""

    if 'ITOP' in config['dataset'] or 'BAD' in config['dataset']:
        try:
            model_name = config.get('model')
            print(model_name)
            if not hasattr(spike, model_name):
                raise ValueError(f"Model {model_name} not found.")

            model_params = get_model_params(config, num_coord_joints)
            Model = getattr(spike, config['model'])
            # Filter the parameters for the specific model
            model_specific_params = {k: v for k, v in model_params.items() if k in config or k == 'num_coord_joints'}
            
            if not model_specific_params:
                raise ValueError(f"No valid parameters found in config for model {model_name}.")

            return Model(**model_specific_params)
        except ValueError as e:
            print(f"Error: {e}")
            return None
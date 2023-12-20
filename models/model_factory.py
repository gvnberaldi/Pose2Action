import models.models_msr as models_msr

def create_model(config, num_classes):
    try:
        model_name = config.get('model')
        if not hasattr(models_msr, model_name):
            raise ValueError(f"Model {model_name} not found in models_msr module.")

        model_params = {
            'P4Transformer': {
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
                'num_classes': num_classes
            },
            'PSTNet': {
                'radius': config.get('radius'),
                'nsamples': config.get('nsamples'),
                'num_classes': num_classes
            },
            'PSTNet2': {
                'radius': config.get('radius'),
                'nsamples': config.get('nsamples'),
                'num_classes': num_classes
            },
            'PSTTransformer': {
                'radius': config.get('radius'),
                'nsamples': config.get('nsamples'),
                'spatial_stride': config.get('spatial_stride'),
                'temporal_kernel_size': config.get('temporal_kernel_size'),
                'temporal_stride': config.get('temporal_stride'),
                'dim': config.get('dim'),
                'depth': config.get('depth'),
                'heads': config.get('heads'),
                'dim_head': config.get('dim_head'),
                'dropout1': config.get('dropout1'),
                'mlp_dim': config.get('mlp_dim'),
                'num_classes': num_classes,
                'dropout2': config.get('dropout2')
            },
            'PrimitiveTransformer': {
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
                'num_classes': num_classes,
            }
        }

        if model_name not in model_params:
            raise ValueError(f"Model {model_name} not found in model_params dictionary.")

        Model = getattr(models_msr, config['model'])

        # Filter the parameters for the specific model
        model_specific_params = {k: v for k, v in model_params[config['model']].items() if k in config or k == 'num_classes'}
        
        if not model_specific_params:
            raise ValueError(f"No valid parameters found in config for model {model_name}.")

        return Model(**model_specific_params)
    except ValueError as e:
            print(f"Error: {e}")
            return None
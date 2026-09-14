"""Derive full Hy4 main/MTP coverage independently of output declarations."""


def full_inventory(config, source_index):
    if (config['num_hidden_layers'], config['num_nextn_predict_layers'],
            config['n_routed_experts']) != (78, 1, 256):
        raise ValueError('This runtime supports the pinned Hy4 78+1/256 layout only')
    logical, physical = set(), set()
    for block in [f'model.layers.{i}' for i in range(1, 78)] + ['model.mtp_layers.0']:
        prefix = block+'.mlp.'
        physical.update(prefix+'experts.'+p for p in ('gate_up_proj', 'down_proj'))
        for p in ('gate_proj', 'up_proj', 'down_proj'):
            shared = prefix+'shared_experts.'+p+'.weight'
            physical.add(shared)
            logical.add(shared)
            logical.update(prefix+f'experts.{e}.'+p+'.weight' for e in range(256))
    keys = set(source_index['weight_map'])
    if not physical <= keys:
        raise ValueError('Source lacks expected main/MTP expert banks')
    retained = keys - physical
    if logical & retained:
        raise ValueError('Ambiguous original parameter names')
    expected = {n: 'quantized' for n in logical}
    expected.update({n: 'retained' for n in retained})
    return expected, physical


def validate_full_state(state, config, source_index):
    expected, physical = full_inventory(config, source_index)
    if state.get('format') != 'hy4_w4a8_v1' or state.get('complete') is not True:
        raise ValueError('Unsupported or incomplete checkpoint')
    if state.get('identity', {}).get('scope') != 'FULL_MAIN_AND_MTP_EXPERTS':
        raise ValueError('A layer candidate is not a complete model')
    if state.get('expected') != expected or set(state.get('parameters', {})) != set(expected):
        raise ValueError('Missing or unexpected main/MTP/retained parameters')
    for name, kind in expected.items():
        if state['parameters'][name].get('kind') != kind:
            raise ValueError('Incorrect parameter precision scope: '+name)
    return expected, physical

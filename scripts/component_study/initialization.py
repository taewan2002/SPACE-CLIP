"""Match the control's shared weights to the full decoder initialization."""


def copy_shared_initialization(full, control):
    source, target = full.state_dict(), control.state_dict()
    allowed_slices = {f"main_decoder_blocks.{i}.fusion_conv.0.weight" for i in range(3)}
    shared, sliced = [], []
    for name, tensor in target.items():
        if name not in source:
            raise ValueError(f"Control tensor has no full-model counterpart: {name}")
        original = source[name]
        if tensor.shape == original.shape:
            target[name] = original.clone()
            if not name.startswith("clip_vision_model."):
                shared.append(name)
        elif (
            name in allowed_slices
            and tensor.ndim == original.ndim == 4
            and tensor.shape[0] == original.shape[0]
            and tensor.shape[2:] == original.shape[2:]
            and tensor.shape[1] < original.shape[1]
        ):
            # Fusion concatenates upsampled, semantic, then structural channels.
            target[name] = original[:, : tensor.shape[1]].clone()
            sliced.append(name)
        else:
            raise ValueError(f"Unexpected control shape: {name}: {tensor.shape}, {original.shape}")
    if set(sliced) != allowed_slices:
        raise ValueError("Expected three fusion-input slices for the early-layer control")
    control.load_state_dict(target, strict=True)
    return {"shared_tensors_identical": shared, "fusion_input_slices": sliced}

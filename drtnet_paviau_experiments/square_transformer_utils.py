from __future__ import annotations

from models.transformer_square_window import TransformerModel_square_window


DEFAULT_COARSE_WINDOW = 8
DEFAULT_FINE_WINDOW = 12


def replace_with_square_window_transformers(
    model,
    coarse_window: int = DEFAULT_COARSE_WINDOW,
    fine_window: int = DEFAULT_FINE_WINDOW,
):
    """Replace DRTNet's rectangle transformers with ordinary square-window ones."""
    model.transformer1 = TransformerModel_square_window(
        map_size=8,
        M_channel=model.n_bands * 2,
        dim=128,
        depth=5,
        heads=8,
        mlp_dim=model.n_bands,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        window_size=coarse_window,
    )
    model.transformer2 = TransformerModel_square_window(
        map_size=32,
        M_channel=model.n_bands,
        dim=64,
        depth=5,
        heads=8,
        mlp_dim=model.n_bands,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        window_size=fine_window,
    )
    model.drtnet_transformer_ablation = "square_window"
    model.drtnet_square_windows = (coarse_window, fine_window)
    return model


def patch_main_square_transformer(
    main_mod,
    coarse_window: int = DEFAULT_COARSE_WINDOW,
    fine_window: int = DEFAULT_FINE_WINDOW,
) -> None:
    base_cls = getattr(main_mod, "MCT_rectangle")

    class SquareWindowDRT(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            replace_with_square_window_transformers(self, coarse_window, fine_window)
            print(
                "Square-window transformer ablation active: "
                "transformer1 window={}x{}, transformer2 window={}x{}".format(
                    coarse_window, coarse_window, fine_window, fine_window
                )
            )

    main_mod.MCT_rectangle = SquareWindowDRT

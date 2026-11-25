from .latte import Latte_models


def get_models(args):
    if "SGDiff" in args.model:
        return Latte_models[args.model](
            input_size=args.latent_size,
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            num_frames=args.num_frames,
            learn_sigma=args.learn_sigma,
            extras=args.extras,
            extras_cond=args.stru_cond
        )
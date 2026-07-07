from .context import EvaluationContext
from .image.evaluator import ImageEvaluator
from .toy2d.evaluator import Toy2DEvaluator


def create_evaluator(
    *,
    data_conf,
    eval_conf,
    synth_data_base_seed,
    sde,
    generator_state,
    out_dir,
    device,
    postprocessing0,
    postprocessing1,
    clip_fid=True,
    autocast=None,
):
    context = EvaluationContext(
        data_conf=data_conf,
        eval_conf=eval_conf,
        synth_data_base_seed=synth_data_base_seed,
        sde=sde,
        generator_state=generator_state,
        out_dir=out_dir,
        device=device,
        postprocessing0=postprocessing0,
        postprocessing1=postprocessing1,
        clip_fid=clip_fid,
        autocast=autocast,
    )
    if context.is_2d:
        return Toy2DEvaluator(context)
    return ImageEvaluator(context)

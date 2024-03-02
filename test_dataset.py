import glob
import os
from loguru import logger
import torch
import argparse
from torch import Tensor

from logger import init_logger
from train import get_device
from config import config
from model import ModelParams
from enhance import init_df, save_audio
from evaluation_utils import evaluation_loop
from libdf import DF


def main(args):
    if args.prft:
        compression_dir = args.model_base_dir + '/DG_compression'
        config.load(os.path.join(args.model_base_dir, "config.ini"))
        p = ModelParams()
        df_state = DF(
            sr=p.sr,
            fft_size=p.fft_size,
            hop_size=p.hop_size,
            nb_bands=p.nb_erb,
            min_nb_erb_freqs=p.min_nb_freqs,
        )
        model_dir = compression_dir + '/finetuned/ft_ep_4.pth'
        log_file = os.path.join(compression_dir, 'test_VoiceDemand.log')
        init_logger(file=log_file, level='INFO', model=model_dir)
        model = torch.load(model_dir).to(get_device())
    else:
        model, df_state, suffix = init_df(
            args.model_base_dir,
            post_filter=args.pf,
            log_level=args.log_level,
            log_file="test_VoiceDemand.log",
            config_allow_defaults=True
        )
    assert os.path.isdir(args.dataset_dir)
    sr = ModelParams().sr
    noisy_dir = os.path.join(args.dataset_dir, "noisy")
    clean_dir = os.path.join(args.dataset_dir, "clean")
    assert os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)
    clean_files = sorted(glob.glob(clean_dir + "/*.wav"))
    noisy_files = sorted(glob.glob(noisy_dir + "/*.wav"))
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    def save_audio_callback(cleanfn: str, enh: Tensor):
        save_audio(os.path.basename(cleanfn), enh, sr, output_dir=args.output_dir, suffix=None)

    metrics = evaluation_loop(
        df_state,
        model,
        clean_files,
        noisy_files,
        n_workers=args.metric_workers,
        save_audio_callback=save_audio_callback if args.output_dir is not None else None,
        metrics=["stoi", "composite", "sisdr"],
        csv_path_enh=args.csv_path_enh,
        csv_path_noisy=args.csv_path_noisy,
        noisy_metric=args.compute_noisy_metric,
        sleep_ms=args.sleep_ms,
    )
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-base-dir",
        type=str,
        default="/E/zhaozugang/SpeechEhancement/DeepFilterNet/self_test/train_result/48KHz/tmp/DFNet2_jit_S_Lin_sp",
        help="Model directory containing checkpoints and config. "
    )
    parser.add_argument(
        "--prft",
        type=bool,
        default=False,
        help="Test orginal model or pruned-finetuned(prft) model",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="/E/zhaozugang/Dataset/VoiceBank_Demand_test/48KHz/tmp",  # VoiceBank_Demand_test/NOIZEUS
        help="Directory in which the enhanced audio files will be stored.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default='/E/zhaozugang/Dataset/VoiceBank_Demand_test/48KHz',  # VoiceBank_Demand_test/NOIZEUS
        help="Voicebank Demand Test set directory. Must contain 'noisy' and 'clean'.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logger verbosity. Can be one of (debug, info, error, none)",
    )
    parser.add_argument(
        "--pf",
        help="Post-filter that slightly over-attenuates very noisy sections.",
        action="store_true",
    )
    parser.add_argument(
        "--metric-workers",
        "-w",
        type=int,
        default=8,
        help="Number of worker processes for metric calculation.",
    )
    parser.add_argument(
        "--csv-path-enh",
        type=str,
        default=None,
        help="Path to csv score file containing metrics of enhanced audios.",
    )
    parser.add_argument(
        "--csv-path-noisy",
        type=str,
        default=None,
        help="Path to csv score file containing metrics of noisy audios.",
    )
    parser.add_argument("--compute-noisy-metric", default=True)
    parser.add_argument("--sleep-ms", type=int, default=0)
    args = parser.parse_args()
    main(args)

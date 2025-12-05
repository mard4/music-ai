"""
Script principale per training CLAP.
Utilizza tutti i moduli della folder CLAP.
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional
import json

import torch
import yaml

from clap.config import CLAPConfig, TrainingConfig
from clap.model_handler import CLAPModelHandler, create_clap_model
from clap.dataset import create_dataloaders, create_clap_dataset
from clap.training import CLAPTrainer
from clap.testing import CLAPTester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('clap_training.log')
    ]
)
logger = logging.getLogger(__name__)


def load_clap_config(config_path: Optional[str] = None) -> CLAPConfig:
    """
    Carica configurazione CLAP da file YAML o usa default.

    Args:
        config_path: Percorso al file di configurazione

    Returns:
        CLAPConfig completo
    """
    if config_path and Path(config_path).exists():
        logger.info(f"Loading CLAP configuration from: {config_path}")
        return CLAPConfig.from_yaml(config_path)

    logger.info("Using default CLAP configuration")
    return CLAPConfig()


async def setup_training_environment(config: CLAPConfig):
    """
    Setup ambiente di training con tutti i componenti.

    Args:
        config: Configurazione CLAP

    Returns:
        Tuple di (model_handler, train_loader, val_loader, test_loader, device)
    """
    logger.info("=" * 60)
    logger.info("SETTING UP CLAP TRAINING ENVIRONMENT")
    logger.info("=" * 60)

    # 1. Setup device
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")


    # 3. Crea dataset
    logger.info(f"\nCreating CLAP dataset:")
    logger.info(f"  Target sample rate: {config.audio_processing.target_sample_rate}Hz")
    logger.info(f"  Max duration: {config.audio_processing.max_duration_seconds}s")

    # Crea dataset asincrono
    dataset = await create_clap_dataset(
        use_clean_labels=True,
        target_sample_rate=config.audio_processing.target_sample_rate,
        max_duration=config.audio_processing.max_duration_seconds
    )

    # 4. Setup dataloaders
    logger.info(f"\nSetting up dataloaders:")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Workers: {config.training.num_workers}")

    train_loader, val_loader, test_loader = await create_dataloaders(
        dataset=dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio
    )

    logger.info(f"\nData loaders created:")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    # 5. Setup optimizer
    logger.info(f"\nSetting up optimizer:")
    logger.info(f"  Type: {config.training.optimizer_type}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Weight decay: {config.training.weight_decay}")

    # 2. Setup modello CLAP
    logger.info(f"\nInitializing CLAP model:")
    logger.info(f"  Model: {config.model.model_name}")
    logger.info(f"  Enable fusion: {config.model.enable_fusion}")
    logger.info(f"  Pretrained: {config.model.pretrained}")

    model_handler = create_clap_model(
        config=config.model,
        pretrained=config.model.pretrained,
        model_path=None  # Carica default pre-trained
    )

    model = model_handler.model
    logger.info(f"Model loaded successfully")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config.training.get_optimizer_kwargs()
    )

    # 6. Setup scheduler
    scheduler = None
    if hasattr(config.training, 'scheduler_type'):
        logger.info(f"\nSetting up scheduler:")
        logger.info(f"  Type: {config.training.scheduler_type}")

        if config.training.scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **config.training.scheduler_params
            )
        elif config.training.scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **config.training.scheduler_params
            )

    logger.info("=" * 60)
    logger.info("SETUP COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    return model_handler, model, optimizer, scheduler, train_loader, val_loader, test_loader, device


def save_training_artifacts(
        model_handler: CLAPModelHandler,
        trainer: CLAPTrainer,
        config: CLAPConfig,
        output_dir: Path,
        test_metrics: Optional[dict] = None
):
    """
    Salva tutti gli artifact del training.

    Args:
        model_handler: Handler del modello
        trainer: Trainer CLAP
        config: Configurazione
        output_dir: Directory di output
        test_metrics: Metriche di test opzionali
    """
    logger.info(f"\nSaving training artifacts to: {output_dir}")

    # 1. Salva modello finale
    final_model_path = output_dir / "final_model.pt"
    model_handler.save_model(
        filepath=str(final_model_path),
        include_optimizer=True,
        optimizer=trainer.optimizer,
        epoch=trainer.current_epoch,
        metrics={
            'best_val_loss': trainer.best_val_loss,
            'final_val_accuracy': trainer.val_accuracies[-1] if trainer.val_accuracies else 0,
            'test_metrics': test_metrics or {}
        }
    )

    # 2. Salva configurazione
    config_path = output_dir / "training_config.yaml"
    config.save(str(config_path))

    # 3. Salva statistiche training
    stats = trainer._get_training_stats()
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # 4. Salva metriche di test
    if test_metrics:
        test_path = output_dir / "test_results.json"
        with open(test_path, 'w') as f:
            json.dump(test_metrics, f, indent=2, default=str)

    # 5. Salva info modello
    model_info = model_handler.get_model_info()
    info_path = output_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)

    logger.info(f"Artifacts saved:")
    logger.info(f"  - Model: {final_model_path}")
    logger.info(f"  - Config: {config_path}")
    logger.info(f"  - Stats: {stats_path}")
    if test_metrics:
        logger.info(f"  - Test results: {test_path}")
    logger.info(f"  - Model info: {info_path}")


async def run_training_pipeline(
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        resume_checkpoint: Optional[str] = None,
        output_dir: str = "checkpoints",
        skip_test: bool = False
):
    """
    Pipeline completa di training CLAP.

    Args:
        config_path: Percorso file configurazione
        checkpoint_path: Percorso checkpoint pre-addestrato
        resume_checkpoint: Percorso checkpoint per resume training
        output_dir: Directory per salvare checkpoint
        skip_test: Se saltare test finale
    """
    try:
        # 1. Carica configurazione
        logger.info("=" * 60)
        logger.info("CLAP TRAINING PIPELINE")
        logger.info("=" * 60)

        config = load_clap_config(config_path)

        # 2. Crea directory output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path}")

        # 3. Resume training se specificato
        if resume_checkpoint and Path(resume_checkpoint).exists():
            logger.info(f"Resuming training from: {resume_checkpoint}")
            # Qui potresti implementare il resume loading

        # 4. Setup ambiente
        model_handler, model, optimizer, scheduler, train_loader, val_loader, test_loader, device = \
            await setup_training_environment(config)  # AGGIUNTO AWAIT

        # 5. Crea trainer
        trainer = CLAPTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.training,
            device=device
        )

        # 6. Esegui training
        logger.info("\n" + "=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)

        training_stats = trainer.run_training()

        # 7. Test finale
        test_metrics = None
        if not skip_test and test_loader:
            logger.info("\n" + "=" * 60)
            logger.info("FINAL TESTING")
            logger.info("=" * 60)

            tester = CLAPTester(model, device=device)
            test_results = tester.test_model(
                test_loader,
                compute_retrieval=True,
                k_values=config.evaluation.k_values
            )

            test_metrics = {
                'test_loss': test_results['test_loss'],
                'test_accuracy': test_results['test_accuracy'],
                'total_samples': test_results['total_samples'],
                'retrieval_metrics': test_results['retrieval_metrics']
            }

            logger.info(f"\nTest Results:")
            logger.info(f"  Loss: {test_metrics['test_loss']:.4f}")
            logger.info(f"  Accuracy: {test_metrics['test_accuracy']:.4f}")

            if test_metrics['retrieval_metrics']:
                logger.info(f"\nRetrieval Metrics:")
                for key, value in test_metrics['retrieval_metrics'].items():
                    logger.info(f"  {key}: {value:.4f}")

        # 8. Salva artifact
        logger.info("\n" + "=" * 60)
        logger.info("SAVING ARTIFACTS")
        logger.info("=" * 60)

        save_training_artifacts(
            model_handler=model_handler,
            trainer=trainer,
            config=config,
            output_dir=output_path,
            test_metrics=test_metrics
        )

        # 9. Conclusione
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Final epoch: {trainer.current_epoch}")

        if test_metrics:
            logger.info(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")

        logger.info(f"\nArtifacts saved in: {output_path}")

    except Exception as e:
        logger.error(f"\n" + "=" * 60)
        logger.error("TRAINING FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    """Funzione principale con argomenti CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CLAP Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training con configurazione default
  python scripts/train_clap.py

  # Training con configurazione personalizzata
  python scripts/train_clap.py --config config/clap_config.yaml

  # Training con checkpoint specifico
  python scripts/train_clap.py --checkpoint checkpoints/best_model.pt

  # Training con output directory personalizzata
  python scripts/train_clap.py --output-dir runs/experiment_1

  # Training senza test finale
  python scripts/train_clap.py --skip-test
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to CLAP configuration file (YAML)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to pre-trained checkpoint to load"
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints and artifacts"
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip final testing phase"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Imposta livello di log
    logging.getLogger().setLevel(args.log_level)

    # Verifica GPU
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Training will be slower on CPU.")

    # Esegui pipeline
    asyncio.run(run_training_pipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        resume_checkpoint=args.resume,
        output_dir=args.output_dir,
        skip_test=args.skip_test
    ))


if __name__ == "__main__":
    main()
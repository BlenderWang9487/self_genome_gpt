from functools import partial
from pathlib import Path

import datasets
import typer

app = typer.Typer(pretty_exceptions_enable=False)


def _pretrain_generator(fastas: list[Path], max_len: int):
    for fasta in fastas:
        with open(fasta, "r") as f:
            buf = ""
            for line in f:
                if line.startswith(">"):
                    continue
                buf += line.strip().upper()

                while len(buf) >= max_len:
                    yield {"sequence": buf[:max_len]}
                    buf = buf[max_len:]


@app.command()
def pretrain(
    fasta_dir: Path,
    output_dir: Path,
    max_len: int = 1000,
    num_proc: int = 8,
):
    """Pretrain the model on a set of FASTA files."""
    print("get all fasta path...")
    fastas = list(fasta_dir.glob("*.fasta"))
    assert fastas, f"No FASTA files found in {fasta_dir}"

    print("build dataset...")
    generator = partial(_pretrain_generator, max_len=max_len)
    dataset = datasets.Dataset.from_generator(
        generator,
        gen_kwargs={"fastas": fastas},
        num_proc=num_proc,
    )
    dataset.set_format("np")
    dataset.save_to_disk(output_dir)


@app.command()
def placeholder():
    pass


if __name__ == "__main__":
    app()

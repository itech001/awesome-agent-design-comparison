"""Validate a JSON file against the question or result schema."""
from __future__ import annotations

import json
from pathlib import Path

import click
import jsonschema

_HERE = Path(__file__).resolve().parent
_SCHEMAS = {
    "question": _HERE / "question.schema.json",
    "result": _HERE / "result.schema.json",
}


def _load_schema(schema_name: str) -> dict:
    try:
        path = _SCHEMAS[schema_name]
    except KeyError as exc:
        raise ValueError(f"unknown schema: {schema_name}") from exc
    return json.loads(path.read_text())


def validate_file(file_path: str | Path, *, schema_name: str) -> bool:
    """Validate a JSON file. Returns True if valid; raises on invalid."""
    schema = _load_schema(schema_name)
    data = json.loads(Path(file_path).read_text())
    jsonschema.validate(instance=data, schema=schema)
    return True


@click.command()
@click.option("--file", "file_path", required=True, type=click.Path(exists=True))
@click.option("--schema", "schema_name", required=True, type=click.Choice(list(_SCHEMAS)))
def cli(file_path: str, schema_name: str) -> None:
    """Validate FILE against SCHEMA (question | result). Exit 0 if valid."""
    try:
        validate_file(file_path, schema_name=schema_name)
    except jsonschema.ValidationError as exc:
        click.echo(f"INVALID: {exc.message}", err=True)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(2)
    click.echo("OK")


if __name__ == "__main__":
    cli()

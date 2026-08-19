import click
import time
import os
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def cli():
    """ColonyML — Zero-config distributed ML training for CPU clusters."""
    pass


@cli.command()
@click.option("--port", default=29500, help="Port to use (default: 29500)")
def join(port):
    """Join the ColonyML cluster on this network."""

    from colonyml.discovery import NodeDiscovery

    console.print(Panel(
        "[bold cyan]ColonyML[/bold cyan] — Joining cluster...\n"
        "[dim]Machines on the same network will find you automatically.[/dim]",
        border_style="cyan",
        title="ColonyML v0.1.5"
    ))

    discovery = NodeDiscovery(port=port)
    discovery.announce()
    discovery.start_listening()

    def on_found(node):
        console.print(
            f"[green]+ Node joined:[/green] "
            f"[bold]{node['hostname']}[/bold] "
            f"({node['ip']}) | {node['cores']} cores"
        )

    def on_lost(node):
        console.print(
            f"[red]- Node left:[/red] "
            f"[bold]{node['hostname']}[/bold] ({node['ip']})"
        )

    discovery.on_node_found(on_found)
    discovery.on_node_lost(on_lost)

    console.print(
        f"[green]Node announced:[/green] "
        f"[bold]{discovery.hostname}[/bold] "
        f"({discovery.local_ip}:{port}) "
        f"| {discovery._get_cpu_cores()} cores"
    )
    console.print(
        "[dim]Waiting for other nodes — "
        "press Ctrl+C to leave cluster[/dim]\n"
    )

    try:
        while True:
            time.sleep(5)
            nodes = discovery.get_all_nodes()

            table = Table(
                title=f"Cluster — {len(nodes)} node(s)",
                show_header=True,
                header_style="bold cyan"
            )
            table.add_column("Hostname", style="cyan")
            table.add_column("IP Address", style="blue")
            table.add_column("Cores", justify="right")
            table.add_column("Status", justify="center")

            for node in nodes:
                tag = "[green]YOU[/green]" \
                    if node.get("is_self") else "[dim]ready[/dim]"
                table.add_row(
                    node["hostname"],
                    node["ip"],
                    str(node["cores"]),
                    tag
                )

            console.clear()
            console.print(table)

    except KeyboardInterrupt:
        console.print("\n[yellow]Leaving cluster...[/yellow]")
        discovery.stop()
        console.print("[dim]Goodbye.[/dim]")


@cli.command()
def status():
    """Show this machine's current resource status."""

    cpu_cores = os.cpu_count() or 1
    cpu_usage = psutil.cpu_percent(interval=0.5)
    cpu_freq = psutil.cpu_freq()
    ram = psutil.virtual_memory()
    freq_mhz = cpu_freq.current if cpu_freq else 0

    console.print(Panel(
        f"[bold]Hostname:[/bold]      {os.environ.get('COMPUTERNAME', 'unknown')}\n"
        f"[bold]CPU Cores:[/bold]     {cpu_cores}\n"
        f"[bold]CPU Usage:[/bold]     {cpu_usage:.1f}%\n"
        f"[bold]CPU Frequency:[/bold] {freq_mhz:.0f} MHz\n"
        f"[bold]RAM Available:[/bold] "
        f"{ram.available / (1024**3):.1f} GB / "
        f"{ram.total / (1024**3):.1f} GB\n"
        f"[bold]RAM Usage:[/bold]     {ram.percent:.1f}%",
        title="[cyan]ColonyML Node Status[/cyan]",
        border_style="green"
    ))


@cli.command()
def version():
    """Show ColonyML version."""
    from colonyml import __version__
    console.print(f"ColonyML v{__version__}")


@cli.command()
@click.option("--script", required=True,
              help="Path to training script")
@click.option("--epochs", default=3,
              help="Number of epochs (default: 3)")
@click.option("--batch-size", default=64,
              help="Total batch size (default: 64)")
@click.option("--wait", default=10,
              help="Seconds to wait for other nodes (default: 10)")
@click.option("--port", default=29500,
              help="Port to use (default: 29500)")
def train(script, epochs, batch_size, wait, port):
    """
    Start distributed training across all nodes on the network.

    Run the SAME command on every machine:

        colonyml train --script train_mnist.py

    Nodes discover each other automatically.
    No IP addresses. No rank. No config files.
    """
    import importlib.util

    console.print(Panel(
        f"[bold cyan]ColonyML[/bold cyan] — Distributed Training\n"
        f"[dim]Script:  {script}[/dim]\n"
        f"[dim]Epochs:  {epochs}[/dim]\n"
        f"[dim]Batch:   {batch_size}[/dim]\n"
        f"[dim]Wait:    {wait}s for other nodes[/dim]",
        title="ColonyML Train",
        border_style="cyan"
    ))

    if not os.path.exists(script):
        console.print(
            f"[red]Error: Script not found: {script}[/red]"
        )
        return

    # Load the training script as a module
    spec = importlib.util.spec_from_file_location(
        "train_script", script
    )
    module = importlib.util.module_from_spec(spec)

    # Inject ColonyTrainer into script namespace
    from colonyml.trainer import ColonyTrainer
    module.ColonyTrainer = ColonyTrainer
    module.COLONYML_EPOCHS = epochs
    module.COLONYML_BATCH_SIZE = batch_size
    module.COLONYML_WAIT = wait
    module.COLONYML_PORT = port

    spec.loader.exec_module(module)

    # Call train_colonyml() if defined in the script
    if hasattr(module, "train_colonyml"):
        module.train_colonyml(
            epochs=epochs,
            batch_size=batch_size,
            wait=wait,
            port=port
        )
    else:
        console.print(
            "[red]Error: Script must define:[/red]\n"
            "[yellow]def train_colonyml("
            "epochs, batch_size, wait, port):[/yellow]"
        )


if __name__ == "__main__":
    cli()
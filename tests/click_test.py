import click

# === Define your two distinct functions ===

def function_one(name):
    """Performs the first specific task using the name."""
    click.echo("-" * 20)
    click.echo("Executing Function One...")
    click.echo(f"Task one processing for: {name}")
    # Imagine more complex logic here
    click.echo("Function One finished.")
    click.echo("-" * 20)

def function_two(count, name):
    """Performs the second specific task using the count and name."""
    click.echo("-" * 20)
    click.echo("Executing Function Two...")
    click.echo(f"Task two processing {count} times for {name}.")
    for i in range(count):
        click.echo(f"  Iteration {i+1} for {name}")
    # Imagine more complex logic here
    click.echo("Function Two finished.")
    click.echo("-" * 20)


# === Define the single Click command entry point ===

@click.command()
@click.option('--name', default='Default User', help='The name to be used by the functions.')
@click.option('--count', default=1, type=int, help='The count to be used by function_two.')
def cli_entry_point(name, count):
    """
    This command runs Function One and Function Two sequentially.
    """
    click.echo(f"Starting script. Received Name='{name}', Count={count}")
    click.echo("="*30)

    # Call the first distinct function, passing relevant options
    function_one(name=name)

    # Call the second distinct function, passing relevant options
    function_two(count=count, name=name)

    click.echo("="*30)
    click.echo("Script finished.")


# === Standard Python entry point check ===
if __name__ == '__main__':
    cli_entry_point()
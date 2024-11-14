# decorators.py

def count_constraints(num_constraints):
    """
    Decorator to annotate constraint functions with the number of constraints they return.

    Args:
        num_constraints (int): Number of constraints the function returns.

    Returns:
        callable: The decorated function.
    """
    def decorator(func):
        func.num_constraints = num_constraints
        return func
    return decorator

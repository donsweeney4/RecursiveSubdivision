import timeit

def cpu_intensive_task(n):
    """
    Performs a CPU-intensive task: calculating the sum of squares up to n.
    """
    total = 0
    for i in range(n):
        total += i*i
    return total

if __name__ == "__main__":
    n = 100_000_000  # A large number to ensure the task takes a noticeable amount of time
    
    # Use timeit to run the task multiple times and get a reliable average
    # The 'number' parameter specifies how many times to run the function
    print(f"Starting CPU benchmark for n = {n:,}...")
    
    # Run the function once to avoid any initial JIT (Just-In-Time) compilation overhead
    cpu_intensive_task(n)
    
    # Time the execution
    execution_time = timeit.timeit(lambda: cpu_intensive_task(n), number=3)
    
    print(f"Finished. Average execution time: {execution_time / 3:.4f} seconds")
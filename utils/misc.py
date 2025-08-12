def chunk_list(input_list, chunk_size: int):
    """
    Split a list (or a dataframe) into chunks of max size
    
    Params
        @input_list: A list or a dataframe
        @chunk_size: The maximum size of the chunk
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def flatten_list(xss):
    return [x for xs in xss for x in xs]
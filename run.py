import boto3
import time
import json
import csv
import os
import random
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='model_benchmark.log'
)
logger = logging.getLogger('model_benchmark')

# Global timeout setting (will be set by user input)
MAX_TIMEOUT = 30

# Initialize Bedrock client with no retries and user-defined timeout
def initialize_bedrock_client(timeout=30):
    global bedrock_runtime
    bedrock_runtime = boto3.client(
        'bedrock-runtime',
        config=boto3.session.Config(
            connect_timeout=timeout,
            read_timeout=timeout,
            retries={'max_attempts': 0}
        )
    )


# Models to test
MODELS = [
    "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "eu.amazon.nova-lite-v1:0",
    "amazon.titan-embed-text-v2:0",
    "cohere.rerank-v3-5:0"
]

# Load questions and bible text
def load_questions():
    with open('questions.txt', 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Load bible text to add to each request (approximately 6k tokens)
    with open('bible.txt', 'r') as f:
        bible_text = f.read(24000)  # ~6k tokens (4 chars per token)
        print(f"Loaded {len(bible_text)} characters (~{len(bible_text)//4} tokens) from bible.txt")
    
    # Add bible text to each question
    padded_questions = []
    for q in questions:
        padded_questions.append(f"{q}\n\nFor reference, here is some additional context:\n\n{bible_text}")
    
    return padded_questions

# Function to count tokens (simple approximation)
def count_tokens(text):
    # Rough approximation: 4 chars = 1 token
    return len(text) // 4

# Function to invoke model
def invoke_model(model_id, prompt):
    start_time = time.time()
    error = None
    response_text = ""
    
    print(f"Sending request to {model_id}...")
    print(f"Prompt length: {len(prompt)} characters (~{len(prompt)//4} tokens)")
    
    try:
        if "claude" in model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            response_text = response_body['content'][0]['text']
            
        elif "nova" in model_id:
            body = json.dumps({
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]}
                ]
                # Nova doesn't accept max_tokens parameter
            })
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            response_text = response_body['output']
            
        elif "titan-embed" in model_id:
            body = json.dumps({
                "inputText": prompt
            })
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            response_text = str(response_body['embedding'])[:100] + "..."  # Just show part of embedding
            
        elif "rerank" in model_id:
            # Get random chunks from bible.txt for documents
            with open('bible.txt', 'r') as f:
                bible_text = f.read()
            
            # Create 5 random chunks of approximately 1000 characters each
            bible_chunks = []
            bible_length = len(bible_text)
            for _ in range(5):
                start_idx = random.randint(0, bible_length - 1000)
                chunk = bible_text[start_idx:start_idx + 1000].replace('\n', ' ').strip()
                bible_chunks.append(chunk)
            
            body = json.dumps({
                "api_version": 2,  # Updated to minimum required version
                "documents": bible_chunks,
                "query": prompt,
                "top_n": 3
            })
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            response_text = str(response_body['results'])[:100] + "..."
    
    except Exception as e:
        error = str(e)
        logger.error(f"Error invoking {model_id}: {error}")
        print(f"ERROR: Failed to invoke {model_id}: {error}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Log if the request timed out
    if duration >= MAX_TIMEOUT:
        logger.warning(f"Request to {model_id} reached timeout limit of {MAX_TIMEOUT}s")
        print(f"WARNING: Request reached timeout limit of {MAX_TIMEOUT}s")
    
    return {
        "duration": duration,
        "error": error,
        "response": response_text,
        "input_tokens": count_tokens(prompt),
        "output_tokens": count_tokens(response_text) if response_text else 0
    }


# Function to get benchmark configuration
def get_benchmark_config():
    global MAX_TIMEOUT
    
    print("\n=== Benchmark Configuration ===")
    
    # Get timeout setting
    try:
        timeout_input = input(f"Maximum timeout in seconds (default: 30): ")
        if timeout_input.strip():
            MAX_TIMEOUT = int(timeout_input)
            if MAX_TIMEOUT < 1:
                print("Using minimum timeout of 1 second")
                MAX_TIMEOUT = 1
        print(f"Using timeout: {MAX_TIMEOUT} seconds")
    except ValueError:
        print("Invalid input. Using default timeout of 30 seconds.")
        MAX_TIMEOUT = 30
    
    # Initialize Bedrock client with the specified timeout
    initialize_bedrock_client(MAX_TIMEOUT)
    
    # Get iterations for each model
    iterations = {}
    print("\nFor each model, specify how many times to run each question.")
    print("Enter 0 to skip a model completely.")
    
    for model_id in MODELS:
        model_name = model_id.split('.')[1] if '.' in model_id else model_id
        try:
            iterations[model_id] = int(input(f"Number of iterations for {model_name} [{model_id}] (default: 1): ") or "1")
            if iterations[model_id] < 0:
                print("Invalid value. Using default of 1 iteration.")
                iterations[model_id] = 1
            elif iterations[model_id] == 0:
                print(f"Skipping model: {model_id}")
        except ValueError:
            print("Invalid input. Using default of 1 iteration.")
            iterations[model_id] = 1
    
    return iterations

# Main benchmark function
def run_benchmark():
    questions = load_questions()
    results = []
    
    # Get benchmark configuration
    model_iterations = get_benchmark_config()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"results/benchmark_{timestamp}.csv"
    
    print("\n=== Benchmark started ===")
    
    # Write CSV header
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_id", "question_id", "iteration", "duration", "input_tokens", 
            "output_tokens", "total_tokens", "tokens_per_minute", "success"
        ])
    
    # Run benchmark for each model and question
    for model_id in MODELS:
        iterations = model_iterations[model_id]
        
        # Skip models with 0 iterations
        if iterations == 0:
            logger.info(f"Skipping model: {model_id}")
            continue
            
        logger.info(f"Testing model: {model_id} with {iterations} iterations per question")
        model_results = []
        
        for i, question in enumerate(questions):
            logger.info(f"  Question {i+1}/{len(questions)}")
            
            for iteration in range(iterations):
                print(f"\nRunning {model_id} - Question {i+1}/{len(questions)} - Iteration {iteration+1}/{iterations}")
                
                result = invoke_model(model_id, question)
                success = result["error"] is None
                
                # Calculate total tokens and tokens per minute
                total_tokens = result["input_tokens"] + result["output_tokens"]
                tokens_per_minute = (total_tokens / result["duration"]) * 60 if result["duration"] > 0 else 0
                
                # Log result
                logger.info(f"    Iteration {iteration+1}: Duration: {result['duration']:.2f}s, Tokens/min: {tokens_per_minute:.2f}, Success: {success}")
                print(f"  Duration: {result['duration']:.2f}s, Tokens/min: {tokens_per_minute:.2f}, Success: {success}")
                print(f"  Tokens sent: {result['input_tokens']}, Tokens received: {result['output_tokens']}")
                
                # Save to results list
                result_row = {
                    "model_id": model_id,
                    "question_id": i+1,
                    "iteration": iteration+1,
                    "duration": result["duration"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": total_tokens,
                    "tokens_per_minute": tokens_per_minute,
                    "success": success
                }
                model_results.append(result_row)
                
                # Write to CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        model_id, i+1, iteration+1, result["duration"], result["input_tokens"],
                        result["output_tokens"], total_tokens, tokens_per_minute, success
                    ])
                
                # Small delay between requests
                time.sleep(1)
        
        # Only add results if we have any (in case we skipped the model)
        if model_results:
            results.extend(model_results)
    
    logger.info(f"Benchmark complete. Results saved to {csv_file}")
    return csv_file, results

if __name__ == "__main__":
    try:
        csv_file, results = run_benchmark()
        print(f"\nBenchmark complete. Results saved to {csv_file}")
        print("Run 'python generate_graph.py' to create visualization of the results.")
    except Exception as e:
        print(f"\nBenchmark interrupted: {str(e)}")
        print("Partial results may have been saved.")
        print("Run 'python generate_graph.py' to create visualization of any saved results.")
